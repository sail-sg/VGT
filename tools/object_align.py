import sys
sys.path.insert(0, '../')
import h5py
import os.path as osp
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
import sys
sys.path.insert(0, '../')
from util import load_file, save_to
import os

def align_object(video_feature_path, mode):
    bbox_feat_file = osp.join(video_feature_path, 'region_8c10b_{}.h5'.format(mode))
    print('Load {}...'.format(bbox_feat_file))         
    out_file = osp.join(bbox_feat_file+'.h5')
    fout = h5py.File(out_file, 'w')
    string_dt = h5py.special_dtype(vlen=str)
    with h5py.File(bbox_feat_file, 'r') as fp:
        vids = fp['ids']
        feats = fp['feat']
        bboxes = fp['bbox']
        fout.create_dataset('ids', shape=vids.shape, dtype=string_dt, data=vids)
        
        feat_alns, bbox_alns = [], []
        for id, (vid, feat, bbox) in enumerate(zip(vids, feats, bboxes)):
                        
            cnum, fnum, rnum, _ = feat.shape
            cur_feat_aln, cur_bbox_aln = [], []
            for cid, (cur_feat, cur_bbox) in enumerate(zip(feat, bbox)):
                vid_feat_aln, vid_bbox_aln = align(cur_feat, cur_bbox, vid, cid)
                cur_feat_aln.append(vid_feat_aln)
                cur_bbox_aln.append(vid_bbox_aln)
                
            feat_alns.append(cur_feat_aln)
            bbox_alns.append(cur_bbox_aln)
            if id % 100 == 0:
                print(f'{id}/{len(vids)}')

        feat_alns = np.asarray(feat_alns)
        bbox_alns = np.asarray(bbox_alns)
        print(feat_alns.shape, bbox_alns.shape)
        
        fout.create_dataset('feat', shape=feat_alns.shape, dtype=np.float32, data=feat_alns)
        fout.create_dataset('bbox', shape=bbox_alns.shape, dtype=np.float32, data=bbox_alns)


def align_object_byv(video_feature_path, vlist_file):
    vlist = load_file(vlist_file)
    indir = osp.join(video_feature_path, 'bbox_feat')
    outdir = osp.join(video_feature_path, 'bbox_feat_aln')
    vnum = len(vlist)
    print(vnum)
    for idx, vid in enumerate(vlist):
        if idx <= 8000: continue
        if idx > 10000: break
        outfile = osp.join(outdir, vid+'.npz')
        if osp.exists(outfile):
            continue
        infile = osp.join(indir, vid+'.npz')
        region_feat = np.load(infile)
        
        roi_feat, roi_bbox = align_feat_bbox(region_feat['feat'][:8], region_feat['bbox'][:8], vid)
        out_dir = osp.dirname(outfile)
        if not osp.exists(out_dir):
            os.makedirs(out_dir)
        np.savez_compressed(outfile, feat=roi_feat, bbox=roi_bbox)
        if idx % 100 == 0:
            print(f'{idx}/{vnum}', outfile)
            print(roi_feat.shape, roi_bbox.shape)


def align_feat_bbox(feat, bbox, vid):
    cur_feat_aln, cur_bbox_aln = [], []
    for cid, (cur_feat, cur_bbox) in enumerate(zip(feat, bbox)):
        vid_feat_aln, vid_bbox_aln = align(cur_feat, cur_bbox, vid, cid)
        cur_feat_aln.append(vid_feat_aln)
        cur_bbox_aln.append(vid_bbox_aln)
    return np.asarray(cur_feat_aln), np.asarray(cur_bbox_aln)


def align(feats, bboxes, vid, cid):
    new_feats, new_bboxes = [], []
    paths = get_tracks(feats, bboxes, vid, cid)
    for i in range(len(paths)):
        obj_feat, obj_pos = [], []
        for fid in range(len(feats)):
            feat = feats[fid][paths[i][fid]]
            bbox = bboxes[fid][paths[i][fid]]
            obj_feat.append(feat)
            obj_pos.append(bbox)
        new_feats.append(obj_feat)
        new_bboxes.append(obj_pos)
    new_feats = np.asarray(new_feats).transpose(1, 0, 2)
    new_bboxes = np.asarray(new_bboxes).transpose(1, 0, 2)
    return new_feats, new_bboxes


def get_tracks(feats, bboxes, vid, cid):
    links = get_link(feats, bboxes)
    paths = []
    for i in range(bboxes.shape[1]):
        max_path = find_max_path_greedy(links, i)
        links = update_links(links, max_path)
        max_path = [i] + max_path
        paths.append(max_path)
        # vis_path(vid, cid, bboxes, max_path)
        # break
    return paths


def get_link(feats, bboxes):
    fnum = feats.shape[0]
    link_cretiria = []
    for fid in range(fnum-1):
        feat_p, feat_n = feats[fid], feats[fid+1]
        sim_f = pairwise_distances(feat_p, feat_n, 'cosine', n_jobs=1)
        sim_f = 1-sim_f
        box_p, box_n = bboxes[fid], bboxes[fid+1]
        areas_p = np.array([get_area(bbox) for bbox in box_p])
        areas_n = np.array([get_area(bbox) for bbox in box_n])
        op_box = []
        for bid, bbox in enumerate(box_p):
            area_p = areas_p[bid]
            x1 = np.maximum(bbox[0], box_n[:, 0])
            y1 = np.maximum(bbox[1], box_n[:, 1])
            x2 = np.minimum(bbox[2], box_n[:, 2])
            y2 = np.minimum(bbox[3], box_n[:, 3])
            W = np.maximum(0, x2 - x1 + 1)
            H = np.maximum(0, y2 - y1 + 1)
            ov_area = W * H
            IoUs = ov_area / (area_p + areas_n - ov_area)
            op_box.append(IoUs)
        scores = np.asarray(op_box) + sim_f #equal importance
        link_cretiria.append(scores)
    return np.asarray(link_cretiria)


def update_links(links, max_path):
    """
    remove the nodes at the max_path
    """
    for i, v in enumerate(max_path):
        links[i][v] = 0
    return links


def find_max_path_greedy(link_scores, sid):
    path = []
    for i in range(link_scores.shape[0]):
        sid = np.argmax(link_scores[i][sid])
        path.append(sid)
    return path


def get_area(bbox):
     area = (bbox[2]-bbox[0]+1)*(bbox[3]-bbox[1]+1)
     return area


def main():
    video_feature_path = f'../../data/feats/nextqa/region_feat_n/'
    align_object(video_feature_path, 'test')
    # dataset_dir = '../../data/datasets/nextqa/test.csv'
    # vlist_file = dataset_dir + 'vlist.json'
    # if osp.exists(vlist_file):
    #     vlist = load_file(vlist_file)
    # else:
    #     data = load_file(dataset_dir)
    #     vlist = list(set(list(data['video_id'])))
    #     save_to(vlist_file, vlist)
    # align_object_byv(video_feature_path, vlist_file)


if __name__ == "__main__":
    main()
