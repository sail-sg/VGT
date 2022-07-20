# Copyright 2022 Garena Online Private Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
sys.path.insert(0, '../')
from util import tokenize, transform_bb, load_file
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
import pandas as pd
import collections
# from tools.object_align import align
import os.path as osp
import h5py
import random as rd
import numpy as np

class VideoQADataset(Dataset):
    def __init__(
        self,
        csv_path,
        features,
        qmax_words=20,
        amax_words=5,
        bert_tokenizer=None,
        a2id=None,
        ivqa=False,
        max_feats=20,
        mc=0,
        bnum =10
    ):
        """
        :param csv_path: path to a csv containing columns video_id, question, answer
        :param features: dictionary mapping video_id to torch tensor of features
        :param qmax_words: maximum number of words for a question
        :param amax_words: maximum number of words for an answer
        :param bert_tokenizer: BERT tokenizer
        :param a2id: answer to index mapping
        :param ivqa: whether to use iVQA or not
        :param max_feats: maximum frames to sample from a video
        """
        self.csv_path = csv_path
        self.data = pd.read_csv(csv_path)
        self.dset = self.csv_path.split('/')[-2]
        self.all_answers = list(self.data['answer'])
        self.video_feature_path = features
        self.bbox_num = bnum
        self.use_frame = True
        self.use_mot =  False
        self.qmax_words = qmax_words
        self.amax_words = amax_words
        self.a2id = a2id
        self.bert_tokenizer = bert_tokenizer
        self.ivqa = ivqa
        self.max_feats = max_feats
        self.mc = mc
        self.mode = osp.basename(csv_path).split('.')[0] #train, val or test
        if self.dset == 'star':
            self.vid_clips = load_file(osp.dirname(csv_path)+f'/clips_{self.mode}.json')
        
        if self.dset not in ['webvid', 'frameqa']:
            filen = bnum
            if self.dset == 'nextqa': filen = 20
            bbox_feat_file = osp.join(self.video_feature_path, f'region_feat_n/acregion_8c{filen}b_{self.mode}.h5')
            print('Load {}...'.format(bbox_feat_file))          
            self.bbox_feats = {}
            with h5py.File(bbox_feat_file, 'r') as fp:
                vids = fp['ids']
                feats = fp['feat']
                print(feats.shape) #v_num, clip_num, region_per_frame, feat_dim
                bboxes = fp['bbox']
                for id, (vid, feat, bbox) in enumerate(zip(vids, feats, bboxes)):
                    #(clip, frame, bbox, feat), (clip, frame, bbox, coord)
                    if self.dset == 'star': vid = vid.decode("utf-8")
                    self.bbox_feats[str(vid)] = (feat[:,:,:self.bbox_num, :], bbox[:,:,:self.bbox_num, :]) 

        if self.dset not in ['webvid']:   
            if self.use_frame:
                app_feat_file = osp.join(self.video_feature_path, f'frame_feat/app_feat_{self.mode}.h5')
                print('Load {}...'.format(app_feat_file))
                self.frame_feats = {}
                with h5py.File(app_feat_file, 'r') as fp:
                    vids = fp['ids']
                    feats = fp['resnet_features']
                    print(feats.shape) #v_num, clip_num, feat_dim
                    for id, (vid, feat) in enumerate(zip(vids, feats)):
                        self.frame_feats[str(vid)] = feat


    def __len__(self):
        return len(self.data)
    
    def get_video_feature(self, video_name, width=1, height=1):
        """
        :param video_name:
        :param width:
        :param height:
        :return:
        """
        cnum = 8
        cids = list(range(cnum))
        pick_ids = cids
        
        # if self.mode == 'train':
        #     rd.shuffle(pick_ids)
            # pick_ids = sorted(rd.sample(cids, 8))
        
        if self.dset in ['frameqa']:
            region_feat_file = osp.join('../data/feats/TGIF', 'region_feat_aln/'+video_name+'.npz')
            region_feat = np.load(region_feat_file)
            roi_feat, roi_bbox = region_feat['feat'], region_feat['bbox']
        else:
            roi_feat = self.bbox_feats[video_name][0][pick_ids]
            roi_bbox = self.bbox_feats[video_name][1][pick_ids]
        
        bbox_feat = transform_bb(roi_bbox, width, height)
        roi_feat = torch.from_numpy(roi_feat).type(torch.float32)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)

        region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)
        
        if self.use_frame:
            temp_feat = self.frame_feats[video_name][pick_ids] #[:, pick_id,:] #.reshape(clip_num*fnum, -1)[pick_ids,:] #[:,pick_ids,:]
            app_feat = torch.from_numpy(temp_feat).type(torch.float32)
        
        # print('Sampled feat: {}'.format(region_feat.shape))
        return region_feat, app_feat

    def get_video_feat(self, video_name, width=1, height=1):
        video_feature_path = f'../data/feats/{self.dset}/'
        frame_feat_file = osp.join(video_feature_path, 'frame_feat/'+video_name+'.npy')
        frame_feat = np.load(frame_feat_file)
        app_feat = torch.from_numpy(frame_feat).type(torch.float32)
        region_feat_file = osp.join(video_feature_path, 'region_feat_aln/'+video_name+'.npz')
        region_feat = np.load(region_feat_file)
        
        roi_feat, roi_bbox = region_feat['feat'], region_feat['bbox']

        roi_feat = torch.from_numpy(roi_feat).type(torch.float32)

        bbox_feat = transform_bb(roi_bbox, width, height)
        bbox_feat = torch.from_numpy(bbox_feat).type(torch.float32)

        region_feat = torch.cat((roi_feat, bbox_feat), dim=-1)

        return region_feat, app_feat


    def __getitem__(self, index):
        
        cur_sample = self.data.loc[index]
        vid_id = cur_sample["video_id"]
        vid_id = str(vid_id)
        qid = str(cur_sample['qid'])
        if 'width' not in cur_sample:
            #msrvtt
            width, height = 320, 240
        else:
            width, height = cur_sample['width'], cur_sample['height']
        if self.dset == 'webvid':
            video_o, video_f = self.get_video_feat(vid_id, width, height)
        else:
            video_o, video_f = self.get_video_feature(vid_id, width, height)
        
        vid_duration = video_f.shape[0]
        
        question_txt = cur_sample['question']
            
        # print(question_txt)
        if self.qmax_words != 0:
            question_embd = torch.tensor(
                self.bert_tokenizer.encode(
                    question_txt,
                    add_special_tokens=True,
                    padding="longest",
                    max_length=self.qmax_words,
                    truncation=True,
                ),
                dtype=torch.long
            )
            seq_len = torch.tensor([len(question_embd)], dtype=torch.long)
        else:
            question_embd = torch.tensor([0], dtype=torch.long)

        type, answer = 0, 0
        if self.ivqa:
            answer_txt = collections.Counter(
                [
                    self.data["answer1"].values[index],
                    self.data["answer2"].values[index],
                    self.data["answer3"].values[index],
                    self.data["answer4"].values[index],
                    self.data["answer5"].values[index],
                ]
            )
            answer_id = torch.zeros(len(self.a2id))
            for x in answer_txt:
                if x in self.a2id:
                    answer_id[self.a2id[x]] = answer_txt[x]
            answer_txt = ", ".join(
                [str(x) + "(" + str(answer_txt[x]) + ")" for x in answer_txt]
            )
        elif self.mc:
            question_id = vid_id+'_'+qid         
            
            if self.dset=='webvid': # and self.mode == 'train':
                ans = cur_sample["answer"]
                cand_answers = self.all_answers
                answer_txts = rd.sample(cand_answers, self.mc-1)
                answer_txts.append(ans)
                rd.shuffle(answer_txts)
                answer_id = answer_txts.index(ans)
            else:
                answer_id = int(cur_sample["answer"])
                # if 'msrvttmc' in self.csv_path.split('/'):
                #     answer_txt = [self.data["a" + str(i)][index] for i in range(self.mc)]
                # else:
                answer_txts = [question_txt+' [SEP] '+self.data["a" + str(i)][index] for i in range(self.mc)]
                # answer_txts = [qa2description(question_txt, cur_sample['a'+str(i)]) for i in range(self.mc)]
            try:
                answer = tokenize(
                    answer_txts,
                    self.bert_tokenizer,
                    add_special_tokens=True,
                    max_length=self.amax_words,
                    dynamic_padding=True,
                    truncation=True,
                )
                
            except:
                print(answer_txts)
            seq_len = torch.tensor([len(ans) for ans in answer], dtype=torch.long)
        else:
            answer_txts = cur_sample["answer"]
            answer_id = self.a2id.get(answer_txts, -1)  # put an answer_id -1 if not in top answers, that will be considered wrong during evaluation
            
        # if not self.mc:
        #     type = self.data["type"].values[index]
            question_id = qid
       
        return {
            "video_id": vid_id,
            "video_o": video_o,
            "video_f": video_f,
            "video_len": vid_duration,
            "question": question_embd,
            "question_txt": question_txt,
            "type": type,
            "answer_id": answer_id,
            "answer_txt": answer_txts,
            "answer": answer,
            "seq_len": seq_len,
            "question_id": question_id
        }


def videoqa_collate_fn(batch):
    """
    :param batch: [dataset[i] for i in N]
    :return: tensorized batch with the question and the ans candidates padded to the max length of the batch
    """
    qmax_len = max(len(batch[i]["question"]) for i in range(len(batch)))
    
    for i in range(len(batch)):
        if len(batch[i]["question"]) < qmax_len:
            batch[i]["question"] = torch.cat(
                [
                    batch[i]["question"],
                    torch.zeros(qmax_len - len(batch[i]["question"]), dtype=torch.long),
                ],
                0,
            )

    if not isinstance(batch[0]["answer"], int):
        amax_len = max(x["answer"].size(1) for x in batch)
        for i in range(len(batch)):
            if batch[i]["answer"].size(1) < amax_len:
                batch[i]["answer"] = torch.cat(
                    [
                        batch[i]["answer"],
                        torch.zeros(
                            (
                                batch[i]["answer"].size(0),
                                amax_len - batch[i]["answer"].size(1),
                            ),
                            dtype=torch.long,
                        ),
                    ],
                    1,
                )

    return default_collate(batch)


def get_videoqa_loaders(args, features, a2id, bert_tokenizer, test_mode):
    
    if test_mode:
        test_dataset = VideoQADataset(
            csv_path=args.test_csv_path,
            features=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            ivqa=(args.dataset == "ivqa"),
            max_feats=args.max_feats,
            mc=args.mc,
            bnum =args.bnum,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            drop_last=False,
            collate_fn=videoqa_collate_fn,
        )
        train_loader, val_loader = None, None
    else:
        
        train_dataset = VideoQADataset(
        csv_path=args.train_csv_path,
        features=features,
        qmax_words=args.qmax_words,
        amax_words=args.amax_words,
        bert_tokenizer=bert_tokenizer,
        a2id=a2id,
        ivqa=(args.dataset == "ivqa"),
        max_feats=args.max_feats,
        mc=args.mc,
        bnum =args.bnum,
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_thread_reader,
            shuffle=True,
            drop_last=True,
            collate_fn=videoqa_collate_fn,
        )
        if args.dataset.split('/')[0] in ['tgifqa','tgifqa2', 'msrvttmc']:
            args.val_csv_path = args.test_csv_path
        val_dataset = VideoQADataset(
            csv_path=args.val_csv_path,
            features=features,
            qmax_words=args.qmax_words,
            amax_words=args.amax_words,
            bert_tokenizer=bert_tokenizer,
            a2id=a2id,
            ivqa=(args.dataset == "ivqa"),
            max_feats=args.max_feats,
            mc=args.mc,
            bnum =args.bnum,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size_val,
            num_workers=args.num_thread_reader,
            shuffle=False,
            collate_fn=videoqa_collate_fn,
        )
        test_loader = None

    return (train_loader, val_loader, test_loader)
