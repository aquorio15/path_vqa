import base64
import json
import os
import cv2
import pandas as pd
import pickle
import sys
from PIL import Image
import librosa
import numpy as np
import yaml
import torch
import fastnumpyload
import sys
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from timm.data.transforms_factory import create_transform
import torchvision.transforms as T
from torchvision import transforms
from torchvision import datasets
from sklearn.metrics import top_k_accuracy_score
from utils import load_obj_tsv
from torch.nn.utils.rnn import pad_sequence
import re
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import f1_score
from nltk.translate.bleu_score import sentence_bleu


# %%
def exact_div(x, y):
    assert x % y == 0
    return x // y


train_transformations = transforms.Compose(
    [
        transforms.Resize([150, 150]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229]),
    ]
)

FIELDNAMES = ["image_id", "image_w", "image_h", "num_boxes", "boxes", "features"]
SAMPLE_RATE = 8000
HOP_LENGTH = 160
CHUNK_LENGTH = 5
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)

with open(
    "/nfsshare/Amartya/Pathological_Question_Answering/code_new/config.yaml", "r"
) as f:
    my_dict = yaml.load(f, Loader=yaml.Loader)


def load_tsv(split: str):
    tsv_file = (
        "/nfsshare/Amartya/Pathological_Question_Answering/pvqa/images/%s.csv" % (split)
    )
    df = pd.read_csv(tsv_file, delimiter="\t", names=FIELDNAMES)

    data = []
    for i in range(df.shape[0]):
        datum = {}
        datum["img_id"] = "%s_%04d" % (split, df["image_id"][i])
        datum["img_w"] = df["image_w"][i]
        datum["img_h"] = df["image_h"][i]
        datum["num_boxes"] = df["num_boxes"][i]

        boxes = df["boxes"][i]
        buf = base64.b64decode(boxes[1:])
        temp = np.frombuffer(buf, dtype=np.float64).astype(np.float32)
        datum["boxes"] = temp.reshape(datum["num_boxes"], -1)

        features = df["features"][i]
        buf = base64.b64decode(features[1:])
        temp = np.frombuffer(buf, dtype=np.float32)
        datum["features"] = temp.reshape(datum["num_boxes"], -1)

        data.append(datum)

    return data


class PVQADataset:

    def __init__(self, splits: str, data_dir: str, lang: str, feat_dir: str):
        self.name = splits
        self.splits = splits.split(",")
        self.data_dir = data_dir
        self.lang = lang
        self.feat_dir = feat_dir
        # self.examples = open(os.path.join(data_dir, self.splits+'.txt'), 'r').readlines()
        # loading dataset
        
        self.data = []
        for split in self.splits:
            self.data.extend(
                pickle.load(open(f"{my_dict['dir']}/%s_vqa_2800_shuffle1.pkl" % split, "rb"))
            )
        print("Load %d data from splits %s" % (len(self.data), self.name))
        # Convert list to dict for evaluation
        self.id2datum = {datum["question_id"]: datum for datum in self.data}
        self.ans2label = pickle.load(
            open(f"{my_dict['dir']}/trainval_ans2label_2800_shuffle1.pkl", "rb")
        )
        self.label2ans = pickle.load(
            open(f"{my_dict['dir']}/trainval_label2ans_2800_shuffle1.pkl", "rb")
        )

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


def build_transform(input_size, is_train):
    resize_im = input_size > 224
    if is_train:
        # this should always dispatch to transforms
        transform = create_transform(
            input_size=224,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = T.RandomCrop(input_size, padding=4)
        return transform

    t = []
    if resize_im:
        # size = int((256 / 224) * args.input_size)
        size = int((1.0 / 0.96) * input_size)
        t.append(
            transforms.Resize(
                size, interpolation=3
            ),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    return transforms.Compose(t)


class PVQATorchDataset(Dataset):
    def __init__(self, dataset: PVQADataset, transform=train_transformations):
        super(PVQATorchDataset, self).__init__()
        self.raw_dataset = dataset
        self.transforms = transform
        self.transforms = build_transform(input_size=224, is_train=True)
        self.PILimage = T.ToPILImage()
        # loading detection features to img_data
        # self.imgid2img = {}
        # for split in dataset.splits:
        #     data = load_tsv(split)

        #     for datum in data:
        #         self.imgid2img[datum["img_id"]] = datum

        self.data = []
        for datum in self.raw_dataset.data:
            # if datum["img_id"] in self.imgid2img:
            self.data.append(datum)
        print("use %d data in torch dataset" % (len(self.data)))

    def pad_or_trim(self, array, length: int = N_SAMPLES, *, axis: int = -1):

        if torch.is_tensor(array):
            if array.shape[axis] > length:
                array = array.index_select(
                    dim=axis, index=torch.arange(length, device=array.device)
                )

            if array.shape[axis] < length:
                pad_widths = [(0, 0)] * array.ndim
                pad_widths[axis] = (0, length - array.shape[axis])
                array = F.pad(
                    array, [pad for sizes in pad_widths[::-1] for pad in sizes]
                )
        else:
            if array.shape[axis] > length:
                array = array.take(indices=range(length), axis=axis)

            if array.shape[axis] < length:
                pad_widths = [(0, 0)] * array.ndim
                pad_widths[axis] = (0, length - array.shape[axis])
                array = np.pad(array, pad_widths)

        return array

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]
        img_id = datum["img_id"]
        ques_id = datum["question_id"]
        ques = datum["sent"]
        # audio_feat, _ = librosa.load(
        #     os.path.join(
        #         self.raw_dataset.data_dir,
        #         "Audio",
        #         self.raw_dataset.lang,
        #         self.raw_dataset.name,
        #         f"{self.raw_dataset.lang}_{item}.wav",
        #     ),
        #     sr=SAMPLE_RATE,
        # )
        # audio_feat = self.pad_or_trim(audio_feat)
        # audio_feat = torch.tensor(audio_feat).unsqueeze(0).T
        # print(self.raw_dataset.data_dir)
        if ''.join(list(self.raw_dataset.splits[0])[0:4]) == 'test':
            audio_feat = torch.tensor(
                fastnumpyload.load(
                    os.path.join(
                        self.raw_dataset.feat_dir,
                        'features',
                        'hubert',
                        self.raw_dataset.lang,
                        'test',
                        f"{self.raw_dataset.lang}_{item+1}.npy",
                    )
                )
            )
            image = torch.tensor(
            fastnumpyload.load(
                os.path.join(
                    self.raw_dataset.feat_dir,
                    'features/'
                    "images",
                    "clip",
                    'test',
                    f"{img_id}.npy.npy",
                )
            )
            )
            text_emb = torch.tensor(
                fastnumpyload.load(
                    os.path.join(
                        self.raw_dataset.feat_dir,
                        'features',
                        'text_embeddings',
                        'clip',
                        self.raw_dataset.lang,
                        'test',
                        f"{self.raw_dataset.lang}_{item+1}.npy",
                    )
                )
            )
            ans_text_emb = torch.tensor(
                fastnumpyload.load(
                    os.path.join(
                        self.raw_dataset.feat_dir,
                        'features',
                        'text_embeddings',
                        'clip',
                        self.raw_dataset.lang,
                        'answer',
                        'test',
                        f"{self.raw_dataset.lang}_{item+1}.npy",
                    )
                )
            )
        else:
            audio_feat = torch.tensor(
                fastnumpyload.load(
                    os.path.join(
                        self.raw_dataset.feat_dir,
                        'features',
                        'hubert',
                        self.raw_dataset.lang,
                        self.raw_dataset.name,
                        f"{self.raw_dataset.lang}_{item+1}.npy",
                    )
                )
            )
            text_emb = torch.tensor(
                fastnumpyload.load(
                    os.path.join(
                        self.raw_dataset.feat_dir,
                        'features',
                        'text_embeddings',
                        'clip',
                        self.raw_dataset.lang,
                        self.raw_dataset.name,
                        f"{self.raw_dataset.lang}_{item+1}.npy",
                    )
                )
            )
            image = torch.tensor(
            fastnumpyload.load(
                os.path.join(
                    self.raw_dataset.feat_dir,
                    'features/'
                    "images",
                    "clip",
                    self.raw_dataset.name,
                    f"{img_id}.npy.npy",
                )
            )
            )
            ans_text_emb = torch.tensor(
                fastnumpyload.load(
                    os.path.join(
                        self.raw_dataset.feat_dir,
                        'features',
                        'text_embeddings',
                        'clip',
                        self.raw_dataset.lang,
                        'answer',
                        self.raw_dataset.name,
                        f"{self.raw_dataset.lang}_{item+1}.npy",
                    )
                )
            )
        # audio_feat = torch.tensor(fastnumpyload.load(
        #     os.path.join(
        #         self.raw_dataset.data_dir,
        #         "features",
        #         'Hubert',
        #         self.raw_dataset.lang,
        #         self.raw_dataset.name,
        #         f"{self.raw_dataset.lang}_{item+1}.npy",
        #     )
        # ).squeeze())
            # image = self.transforms(
            #     Image.open(
            #         os.path.join(
            #             self.raw_dataset.data_dir,
            #             "pvqa",
            #             "images",
            #             self.raw_dataset.name,
            #             f"{img_id}.jpg",
            #         )
            #     ).convert("RGB")
            # )
        
        if not image.shape[0] == 0:
            if "label" in datum:
                label = datum["label"]
                target = torch.zeros(self.raw_dataset.num_answers)
                for ans, score in label.items():
                    if ans in self.raw_dataset.ans2label:
                        target[self.raw_dataset.ans2label[ans]] = score
                    return (
                        ques_id,
                        audio_feat,
                        text_emb,
                        image,
                        ans_text_emb,
                        target,
                    )
            else:
                return ques_id, audio_feat, text_emb, image, ans_text_emb 


def collate_function(batch_data):
    batch_data = list(filter(lambda x: x is not None, batch_data))
    ques_id, audio_features, text_emb, image, ans_text_emb, target = zip(*batch_data)
    audio_feature = pad_sequence(audio_features, batch_first=True, padding_value=0)
    text_emb = pad_sequence(text_emb, batch_first=True, padding_value=0)
    # [print(x.size()) for x in image]
    ans_text_emb = pad_sequence(ans_text_emb, batch_first=True, padding_value=0)
    image = pad_sequence(image, batch_first=True, padding_value=0)
    # print(audio_feature.shape)
    return ques_id, audio_features, text_emb, image, ans_text_emb, target


question_types = (
    "where",
    "what",
    "how",
    "how many/how much",
    "when",
    "why",
    "who/whose",
    "other",
    "yes/no",
)


def get_q_type(q: str):
    q = q.lower()
    if q.startswith("how many") or q.startswith("how much"):
        return "how many/how much"
    first_w = q.split()[0]
    if first_w in ("who", "whose"):
        return "who/whose"
    for q_type in ("where", "what", "how", "when", "why"):
        if first_w == q_type:
            return q_type
    if first_w in [
        "am",
        "is",
        "are",
        "was",
        "were",
        "have",
        "has",
        "had",
        "does",
        "do",
        "did",
        "can",
        "could",
    ]:
        return "yes/no"
    if "whose" in q:
        return "who/whose"
    if "how many" in q or "how much" in q:
        return "how many/how much"
    for q_type in ("where", "what", "how", "when", "why"):
        if q_type in q:
            return q_type
    print(q)
    return "other"


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    ret = []
    for k in topk:
        correct = (target * torch.zeros_like(target).scatter(1, pred[:, :k], 1)).float()
        ret.append(correct.sum() / target.sum())
    return ret


class PVQAEvaluator:
    def __init__(self, dataset: PVQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict, logit, target):
        score = 0.0
        qtype_score = {qtype: 0.0 for qtype in question_types}
        qtype_cnt = {qtype: 0 for qtype in question_types}
        preds = []
        anss = []
        b_scores = []
        b_scores1 = []
        b_scores2 = []
        b_scores3 = []
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum["label"]
            quest = datum["sent"]

            q_type = get_q_type(my_dict["save_dir"])
            qtype_cnt[q_type] += 1

            hypo = str(ans).lower().split()
            refs = []
            preds.append(self.dataset.ans2label[ans])
            if ans in label:
                score += label[ans]
                qtype_score[q_type] += label[ans]
            ans_flag = True
            for gt_ans in label:
                refs.append(str(gt_ans).lower().split())
                if ans_flag:
                    anss.append(
                        self.dataset.ans2label[gt_ans]
                        if gt_ans in self.dataset.ans2label
                        else -1
                    )
                    ans_flag = False
            b_score = sentence_bleu(references=refs, hypothesis=hypo)
            b_score1 = sentence_bleu(
                references=refs, hypothesis=hypo, weights=[1, 0, 0, 0]
            )
            b_score2 = sentence_bleu(
                references=refs, hypothesis=hypo, weights=[0, 1, 0, 0]
            )
            b_score3 = sentence_bleu(
                references=refs, hypothesis=hypo, weights=[0, 0, 1, 0]
            )

            b_scores.append(b_score)
            b_scores1.append(b_score1)
            b_scores2.append(b_score2)
            b_scores3.append(b_score3)
        b_score_m = np.mean(b_scores)
        b_score_m1 = np.mean(b_scores1)
        b_score_m2 = np.mean(b_scores2)
        b_score_m3 = np.mean(b_scores3)
        anss_tensor = torch.tensor(anss)
        preds_tensor = torch.tensor(preds)
        # acc1, acc5, acc10 = accuracy(logit, target, topk=(1, 5, 10))
        info = "b_score=%.4f\n" % b_score_m
        info += "b_score1 = %.4f\n" % b_score_m1
        info += "b_score2 = %.4f\n" % b_score_m2
        info += "b_score3 = %.4f\n" % b_score_m3
        info += (
            "Top1_score=%.4f\n"
            % (sum(anss_tensor == preds_tensor) / len(preds_tensor)).item()
        )
        info += "Top5_score=%.4f\n" % accuracy(logit, target, topk=(1, 5))[-1]
        info += "Top10_score=%.4f\n" % accuracy(logit, target, topk=(1, 10))[-1]
        info += "f1_score=%.4f\n" % f1_score(anss, preds, average="macro")
        info += "score = %.4f\n" % (score / len(quesid2ans))
        for q_type in question_types:
            if qtype_cnt[q_type] > 0:
                qtype_score[q_type] /= qtype_cnt[q_type]
        info += "Overall score: %.4f\n" % (score / len(quesid2ans))
        for q_type in question_types:
            info += "qtype: %s\t score=%.4f\n" % (q_type, qtype_score[q_type])

        with open(os.path.join(my_dict["save_dir"], "result_by_type.txt"), "a") as f:
            f.write(info)
        print(info)
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        with open(my_dict["save_dir"] + "/results.txt", "w") as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({"question_id": ques_id, "answer": ans})
            json.dump(result, f, indent=4, sort_keys=True)


# NOTE: Check the dataset
# dset = PVQADataset(
#     splits="train",
#     data_dir="/nfsshare/Amartya/Pathological_Question_Answering",
#     feat_dir = '/nfsshare/Amartya/Pathological_Question_Answering',
#     lang="French",
# )
# tset = PVQATorchDataset(dset)
# evaluator = PVQAEvaluator(dset)
# data_loader = DataLoader(
#     tset, batch_size=1, shuffle=False, num_workers=8, drop_last=True, collate_fn=collate_function
# )

# ques_id, audio_feat, image, target = next(iter(data_loader))
# # print(audio_feat.shape)
# # print(image.shape)
# height = 224 # H ("The training resolution is 224.")
# width = 224 # W
# color_channels = 3 # C
# patch_size = 16 # P
# # Calculate N (number of patches)
# number_of_patches = int((height * width) / patch_size**2)
# embedding_layer_input_shape = (height, width, color_channels)
# embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)
# plt.imshow(image.permute(1, 2, 0))
# print(target)
# print(dset.num_answers)

# %%
