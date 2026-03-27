import os
import kagglehub

save_path = os.path.join(os.getcwd(), 'resources', 'dataset')
if not os.path.exists(save_path):
    os.makedirs(save_path)

def check_dataset():
    if not os.path.exists(os.path.join(save_path, 'drone_seg_dataset')):
        print('Dataset folder not found, downloading...')
        os.makedirs(os.path.join(save_path, 'drone_seg_dataset'))
        try:
            kagglehub.dataset_download(
                handle="santurini/semantic-segmentation-drone-dataset",
                output_dir=os.path.join(save_path, 'drone_seg_dataset'),
            )
            print('Dataset downloaded!')
        except Exception as e:
            print(f"Error downloading dataset: {e}")
    else:
        print('Dataset already exists.')

    return os.path.join(save_path, 'drone_seg_dataset')

if __name__ == '__main__':
    check_dataset()