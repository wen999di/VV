from FaceRec import FaceRecognizer
#from FaceRec_insightface import FaceRecognizer
from CutSubtitle import SubtitleExtractor
#from CutSubtitle_paddleocr import SubtitleExtractor
import os
import traceback

def clean_frames_folder(frames_output):
    """清理帧输出文件夹中的所有文件"""
    if os.path.exists(frames_output):
        for file in os.listdir(frames_output):
            file_path = os.path.join(frames_output, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error: {e}")


def process_video(video_path, features_file, frames_output, subtitle_output):
    """处理单个视频并生成字幕"""
    # 步骤 1: 人脸识别和帧提取
    face_recognizer = FaceRecognizer(features_file)
    face_recognizer.process_video_with_params(
        video_path=video_path,
        output_folder=frames_output,
        fps=1
    )

    # 步骤 2: 字幕提取
    subtitle_extractor = SubtitleExtractor()
    subtitle_extractor.process_frames(
        input_folder=frames_output,
        output_folder=subtitle_output
    )


def process_videos_in_folder(videos_folder, features_file, frames_output, subtitle_output):
    """处理文件夹中的所有视频"""
    # 确保输出目录存在
    os.makedirs(frames_output, exist_ok=True)
    os.makedirs(subtitle_output, exist_ok=True)
    
    # 用于跟踪已处理的视频
    processed_videos = set()
    
    while True:
        # 获取所有视频文件
        video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
        video_files = [f for f in os.listdir(videos_folder)
                      if os.path.isfile(os.path.join(videos_folder, f))
                      and f.lower().endswith(video_extensions)
                      and f not in processed_videos]  # 排除已处理的视频

        if not video_files:
            print("\nAll videos processed successfully!")
            break  # 如果没有新视频，直接退出

        print(f"Found {len(video_files)} new videos to process")

        # 处理每个新视频
        for i, video_file in enumerate(video_files, 1):
            video_path = os.path.join(videos_folder, video_file)
            print(f"\nProcessing video {i}/{len(video_files)}: {video_file}")

            try:
                # 处理视频
                process_video(video_path, features_file, frames_output, subtitle_output)
                
                # 添加到已处理列表
                processed_videos.add(video_file)
                
                # 清理帧文件
                clean_frames_folder(frames_output)
                print(f"Cleaned frames for {video_file}")

            except Exception:
                print(f"Error processing {video_file}")
                traceback.print_exc()
                continue


if __name__ == "__main__":
    # 配置路径
    videos_folder = "Videos"  # 包含所有视频的文件夹
    features_file = "face_features.npz"  # 预计算的特征向量文件
    #features_file = "face_features_insightface.npz"  # 预计算的特征向量文件
    frames_output = "output_frames"
    subtitle_output = "subtitle"

    process_videos_in_folder(
        videos_folder=videos_folder,
        features_file=features_file,
        frames_output=frames_output,
        subtitle_output=subtitle_output
    )