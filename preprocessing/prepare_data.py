import os
import argparse
from mp_extraction import annotate_and_save_video
from homo_roi import get_homography_from_images, calculate_square_roi, get_profile_images

def extract_frames(users, hands, wavelengths, cams, frames_per_minute, dump_landmarks, save_failures, start_time=0, end_time=None):
    data_dir = os.path.join("..", "sample_data", "annotated_frames")
    raw_dir = os.path.join("..", "sample_data", "videos")
    for cam in cams:
        for user in users:
            for wavelength in wavelengths:
                for hand in hands:
                    for condition in ["Clean", "Wet", "Warm", "Dirty"]:
                        output_dir = os.path.join(data_dir, cam, user, wavelength, hand, condition)
                        video_path = os.path.join(raw_dir, f"{user}_{condition}_{cam}_{hand}_{wavelength}.mp4")
                        annotate_and_save_video(
                            video_path=video_path,
                            output_dir=output_dir,
                            model_path="hand_landmarker.task",
                            frames_per_minute=frames_per_minute,
                            hand_label=hand,
                            dump_landmarks=dump_landmarks,
                            save_failures=save_failures,
                            start_time=start_time,
                            end_time=end_time,
                        )

def get_homo_roi(users, cams, hands, wavelengths, fraction=100):
    data_dir = os.path.join("..", "sample_data", "annotated_frames")
    for user in users:
        for cam in cams:
            for hand in hands:
                for wavelength in wavelengths:
                    for condition in ["Clean", "Wet", "Warm", "Dirty"]:
                        src_folder = os.path.join(data_dir, cam, user, wavelength, hand, condition)
                        try:
                            profile_index, profile_path = get_profile_images(src_folder)
                        except Exception as err:
                            log_file_path = "error_log.txt"
                            with open(log_file_path, "a") as file:
                                file.write(f"{str(err)}\n")
                            continue
                        for root, dirs, files in os.walk(src_folder):
                            for file in files:
                                if file.endswith('.png') and "corrected" not in file:
                                    try:
                                        frame_index = file.split(".")[0]
                                        get_homography_from_images(
                                            cur_path=os.path.join(src_folder, file),
                                            profile_path=profile_path,
                                            frame_index=frame_index,
                                            profile_index=profile_index,
                                            pkl_file=os.path.join(src_folder, "landmarks_success.pkl"),
                                            landmark_indices=[0, 1, 5, 17],
                                        )
                                        corrected_file = file.replace(".png", "_corrected.png")
                                        calculate_square_roi(
                                            image_path=os.path.join(src_folder, corrected_file),
                                            frame_index=frame_index,
                                            corrected_pkl_path=os.path.join(src_folder, "landmarks_success_corrected.pkl"),
                                            output_dir=os.path.join(data_dir, f"{cam}_ROI", str(fraction), user, wavelength, hand, condition),
                                            fraction=fraction,
                                            visualization=False,
                                            profile=False,
                                        )
                                    except Exception as err:
                                        continue

def main():
    parser = argparse.ArgumentParser(description="Palm vein data processing pipeline")
    parser.add_argument("--users", nargs="+", required=True, help="List of user IDs")
    parser.add_argument("--hands", nargs="+", required=True, help="List of hands (e.g., L, R)")
    parser.add_argument("--cams", nargs="+", required=True, help="List of camera names (e.g., OpenMV, See3)")
    parser.add_argument("--wavelengths", nargs="+", required=True, help="List of wavelengths (e.g., 850, 890)")
    parser.add_argument("--frames_per_minute", type=int, default=10, help="Frames to extract per minute")
    parser.add_argument("--dump_landmarks", action="store_true", help="Dump landmarks to file")
    parser.add_argument("--save_failures", action="store_true", help="Save failed frames")
    parser.add_argument("--start_time", type=int, default=0, help="Start time in seconds")
    parser.add_argument("--end_time", type=int, help="End time in seconds")
    parser.add_argument("--fraction", type=int, default=10, help="ROI fraction")
    args = parser.parse_args()

    extract_frames(
        users=args.users,
        hands=args.hands,
        wavelengths=args.wavelengths,
        cams=args.cams,
        frames_per_minute=args.frames_per_minute,
        dump_landmarks=args.dump_landmarks,
        save_failures=args.save_failures,
        start_time=args.start_time,
        end_time=args.end_time,
    )
    get_homo_roi(
        users=args.users,
        cams=args.cams,
        hands=args.hands,
        wavelengths=args.wavelengths,
        fraction=args.fraction,
    )

if __name__ == "__main__":
    main()
