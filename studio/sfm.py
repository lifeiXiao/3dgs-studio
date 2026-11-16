"""COLMAP Python bindings: SIFT → matching → sparse mapping → undistortion."""
import os
from pathlib import Path
import pycolmap
from .dataset import EXTS


def image_files(root):
    root=Path(root)
    base=root/'images' if (root/'images').is_dir() else root
    files=sorted(p for p in base.rglob('*') if p.suffix.lower() in EXTS and p.is_file() and not p.is_symlink())
    files=[p for p in files if not any(part.startswith('.') for part in p.relative_to(base).parts)]
    if any(not p.resolve().is_relative_to(root.resolve()) for p in files): raise ValueError('图片路径不能越出数据集')
    return base, files


def reconstruct(source, work, output, update, stopped, matching='exhaustive', max_image_size=1600):
    base,files=image_files(source)
    if len(files)<3: raise ValueError('COLMAP 至少需要 3 张有重叠、清晰且有纹理的照片')
    work.mkdir(parents=True,exist_ok=True); output.mkdir(parents=True,exist_ok=True)
    db=work/'database.db'; sparse=work/'sparse'; sparse.mkdir(exist_ok=True)
    def phase(name,step):
        if stopped(): raise InterruptedError('用户停止了 COLMAP 任务')
        update(stage=name,sfm_step=step,sfm_total=4)
        print(f'[COLMAP {step}/4] {name}',flush=True)
    gpu=bool(pycolmap.has_cuda)
    device=pycolmap.Device.cuda if gpu else pycolmap.Device.cpu
    options=pycolmap.FeatureExtractionOptions(); options.max_image_size=max_image_size
    options.num_threads=min(8,os.cpu_count() or 1); options.use_gpu=gpu
    options.sift.max_num_features=4096
    phase('COLMAP · SIFT 特征提取',0)
    update(sfm_device='CUDA' if gpu else 'CPU',input_images=len(files))
    pycolmap.extract_features(str(db),str(base),image_names=[str(p.relative_to(base)) for p in files],
                             camera_model='SIMPLE_RADIAL',extraction_options=options,device=device)
    phase('COLMAP · 特征匹配',1)
    matches=pycolmap.FeatureMatchingOptions(); matches.num_threads=min(8,os.cpu_count() or 1); matches.use_gpu=gpu
    if matching=='sequential': pycolmap.match_sequential(str(db),matching_options=matches,device=device)
    else: pycolmap.match_exhaustive(str(db),matching_options=matches,device=device)
    phase('COLMAP · 稀疏重建 / Bundle Adjustment',2)
    mapper=pycolmap.IncrementalPipelineOptions(); mapper.num_threads=min(8,os.cpu_count() or 1)
    mapper.min_model_size=3
    reconstructions=pycolmap.incremental_mapping(str(db),str(base),str(sparse),options=mapper)
    if not reconstructions:
        raise ValueError('COLMAP 未能重建相机。请增加视角重叠、纹理与视差，避免模糊、纯旋转和反光。')
    best_id=max(reconstructions,key=lambda k:reconstructions[k].num_reg_images())
    best=reconstructions[best_id]
    if best.num_reg_images()<3 or best.num_points3D()<8: raise ValueError('有效相机或稀疏点不足，无法训练')
    phase('COLMAP · 图片去畸变与模型导出',3)
    pycolmap.undistort_images(str(output),str(sparse/str(best_id)),str(base))
    rec=pycolmap.Reconstruction(str(output/'sparse'))
    rec.write_text(str(output/'sparse'))
    update(sfm_step=4,registered_images=best.num_reg_images(),sparse_points=best.num_points3D(),
           unregistered_images=len(files)-best.num_reg_images())
    if stopped(): raise InterruptedError('用户停止了 COLMAP 任务')
    return output
