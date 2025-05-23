import cv2
import numpy as np
import time
import os
import subprocess
from tqdm import tqdm
import tempfile
import shutil


def process_video_with_dual_ocr(input_video_path, output_video_path, v4_model_name, v5_model_name):
    """
    使用两个不同版本的paddleocr命令行工具检测视频中的字幕位置，
    并将两个版本的检测结果左右并排显示在同一个视频中
    
    参数:
        input_video_path: 输入视频路径
        output_video_path: 输出视频路径
    """
    # 创建临时目录用于存储临时图像文件
    temp_dir = tempfile.mkdtemp()
    frames_dir = os.path.join(temp_dir, "frames")
    v4_results_dir = os.path.join(temp_dir, "v4_results")
    v5_results_dir = os.path.join(temp_dir, "v5_results")
    
    # 创建必要的目录
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(v4_results_dir, exist_ok=True)
    os.makedirs(v5_results_dir, exist_ok=True)
    
    try:
        # 打开视频文件
        cap = cv2.VideoCapture(input_video_path)
        if not cap.isOpened():
            print(f"无法打开视频: {input_video_path}")
            return
        
        # 获取视频属性
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"视频信息: {width}x{height}, {fps}fps, 总帧数: {total_frames}")
        
        # 创建视频写入器 - 输出视频宽度是原始宽度的2倍（左右并排）
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width*2, height))
        
        # 处理进度
        frame_count = 0
        start_time = time.time()
        
        # 第一步：提取所有帧并保存到临时目录
        print("步骤1：提取视频帧...")
        with tqdm(total=total_frames, desc="提取视频帧", unit="帧") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # 保存当前帧为临时图像文件
                frame_filename = f"frame_{frame_count:06d}.jpg"
                frame_path = os.path.join(frames_dir, frame_filename)
                cv2.imwrite(frame_path, frame)
                
                # 更新进度条
                pbar.update(1)
        
        # 释放视频捕获资源
        cap.release()
        
        # 第二步：使用PP-OCRv4处理所有帧
        print("\n步骤2：使用PP-OCRv4处理所有帧...")
        cmd_v4 = f"paddleocr text_detection -i {frames_dir} --save_path={v4_results_dir} --device gpu:0 --model_name {v4_model_name}"
        print(f"执行命令: {cmd_v4}")
        
        try:
            subprocess.run(cmd_v4, shell=True, check=True)
            print("PP-OCRv4处理完成")
        except subprocess.CalledProcessError as e:
            print(f"执行PP-OCRv4命令时出错: {e}")
            return
        
        # 第三步：使用PP-OCRv5处理所有帧
        print("\n步骤3：使用PP-OCRv5处理所有帧...")
        cmd_v5 = f"paddleocr text_detection -i {frames_dir} --save_path={v5_results_dir} --device gpu:0 --model_name {v5_model_name}"
        print(f"执行命令: {cmd_v5}")
        
        try:
            subprocess.run(cmd_v5, shell=True, check=True)
            print("PP-OCRv5处理完成")
        except subprocess.CalledProcessError as e:
            print(f"执行PP-OCRv5命令时出错: {e}")
            return
        
        # 第四步：合并两个版本的结果并生成视频
        print("\n步骤4：合成对比视频...")
        
        # 获取所有处理后的图像文件
        processed_frames = []
        for i in range(1, frame_count + 1):
            frame_filename = f"frame_{i:06d}.jpg"
            v4_result_path = os.path.join(v4_results_dir, frame_filename.replace(".jpg", "_res.jpg"))
            v5_result_path = os.path.join(v5_results_dir, frame_filename.replace(".jpg", "_res.jpg"))
            original_path = os.path.join(frames_dir, frame_filename)
            
            # 检查处理后的图像是否存在
            v4_exists = os.path.exists(v4_result_path)
            v5_exists = os.path.exists(v5_result_path)
            
            processed_frames.append((i, original_path, v4_result_path if v4_exists else None, v5_result_path if v5_exists else None))
        
        # 按帧序号排序
        processed_frames.sort(key=lambda x: x[0])
        
        # 合成视频
        with tqdm(total=len(processed_frames), desc="合成对比视频", unit="帧") as pbar:
            for idx, original_path, v4_path, v5_path in processed_frames:
                # 读取原始帧
                original_frame = cv2.imread(original_path)
                
                if original_frame is None:
                    print(f"警告: 无法读取原始图像: {original_path}")
                    continue
                
                # 读取v4处理结果，如果不存在则使用原始帧
                if v4_path and os.path.exists(v4_path):
                    v4_frame = cv2.imread(v4_path)
                    if v4_frame is None:
                        v4_frame = original_frame.copy()
                        print(f"警告: 无法读取V4处理结果: {v4_path}")
                else:
                    v4_frame = original_frame.copy()
                
                # 读取v5处理结果，如果不存在则使用原始帧
                if v5_path and os.path.exists(v5_path):
                    v5_frame = cv2.imread(v5_path)
                    if v5_frame is None:
                        v5_frame = original_frame.copy()
                        print(f"警告: 无法读取V5处理结果: {v5_path}")
                else:
                    v5_frame = original_frame.copy()
                
                # 在左侧v4帧上添加版本标记
                cv2.putText(v4_frame, v4_model_name,
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # 在右侧v5帧上添加版本标记
                cv2.putText(v5_frame, v5_model_name,
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                
                # 水平拼接两个帧
                combined_frame = np.hstack((v4_frame, v5_frame))
                
                # 写入输出视频
                out.write(combined_frame)
                
                # 更新进度条
                pbar.update(1)
                    
    except Exception as e:
        print(f"处理视频时出错: {e}")
    finally:
        # 释放资源
        if 'cap' in locals() and cap is not None:
            cap.release()
        if 'out' in locals() and out is not None:
            out.release()
        cv2.destroyAllWindows()
        
        # 清理临时目录
        print("清理临时文件...")
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        print(f"视频处理完成! 输出文件: {output_video_path}")
        print(f"总耗时: {time.time() - start_time:.2f}秒")


# 批量处理视频文件夹中的所有视频
def batch_process_videos(input_folder, output_folder, v4_model_name, v5_model_name):
    """
    批量处理文件夹中的所有视频
    
    参数:
        input_folder: 输入视频文件夹
        output_folder: 输出视频文件夹
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有视频文件
    video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".wmv"]
    video_files = [f for f in os.listdir(input_folder) 
                  if os.path.isfile(os.path.join(input_folder, f)) and 
                  any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"在 {input_folder} 中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件")
    
    # 使用tqdm创建进度条处理每个视频
    for i, video_file in enumerate(tqdm(video_files, desc="批量处理视频", unit="个"), 1):
        input_path = os.path.join(input_folder, video_file)
        output_filename = f"processed_{video_file}"
        output_path = os.path.join(output_folder, output_filename)
        
        print(f"[{i}/{len(video_files)}] 处理视频: {video_file}")
        process_video_with_dual_ocr(input_path, output_path, v4_model_name, v5_model_name)
        print(f"完成处理: {video_file} -> {output_filename}\n")


# 主函数
def main():
    # # 单个视频处理
    # input_video = "test/test2.mp4"  # 替换为您的输入视频路径
    # output_video = "output/test2_det.mp4"  # 输出视频路径
    
    # # 检查输入视频是否存在
    # if not os.path.exists(input_video):
    #     print(f"输入视频不存在: {input_video}")
    #     return
    
    # # 处理单个视频
    # process_video_with_dual_ocr(input_video, output_video, "PP-OCRv4_server_det", "PP-OCRv5_server_det")
    
    # 如果需要批量处理，取消下面的注释
    input_folder = "test"  # 输入视频文件夹
    output_folder = "output_mobile_det"  # 输出视频文件夹
    batch_process_videos(input_folder, output_folder, "PP-OCRv4_mobile_det", "PP-OCRv5_mobile_det")
    input_folder = "test"  # 输入视频文件夹
    output_folder = "output_server_det"  # 输出视频文件夹
    batch_process_videos(input_folder, output_folder, "PP-OCRv4_server_det", "PP-OCRv5_server_det")


if __name__ == "__main__":
    main()