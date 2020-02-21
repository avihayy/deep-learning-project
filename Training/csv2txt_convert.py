import os
import pandas as pd
import argparse

def get_parent_dir(n=1):
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

def convert_vott_csv_to_yolo(vott_df, name_of_video=None,labeldict=dict(zip(['vehicle'], [0, ])), path='', temp_name= 'data_train_temp.txt', target_name='data_train.txt',
                             abs_path=False,):
    if not 'code' in vott_df.columns:
        vott_df['code'] = vott_df['label'].apply(lambda x: labeldict[x])
    for col in vott_df[['xmin', 'ymin', 'xmax', 'ymax']]:
        vott_df[col] = (vott_df[col]).apply(lambda x: round(x))

    last_image = ''
    txt_file = ''

    for index, row in vott_df.iterrows():
        if not last_image == row['image']:
            if abs_path:
                txt_file += '\n' + row['image_path'] + ' '
            else:
                txt_file += '\n' + os.path.join(path, row['image']) + ' '
            txt_file += ','.join([str(x) for x in (row[['xmin', 'ymin', 'xmax', 'ymax', 'code']].tolist())])
        else:
            txt_file += ' '
            txt_file += ','.join([str(x) for x in (row[['xmin', 'ymin', 'xmax', 'ymax', 'code']].tolist())])
        last_image = row['image']
    file = open(temp_name, "w")
    file.write(txt_file[1:])
    file.close()
    f = open(temp_name, 'r')
    w = open(target_name, 'w')
    for x in f:
        if name_of_video in x[x.find('/frame') + 1:]:
            w.write(x)
    w.close()
    f.close()
    os.remove(temp_name)
    return True


Data_Folder = os.path.join(get_parent_dir(1),'Data')
Frames_Folder = os.path.join(Data_Folder,'Source_Video','Training_Frames')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    parser.add_argument(
        "--Frames_Folder", type=str, default=Frames_Folder,
        help = "Absolute path to the folder of the csv file and the tagged frames. Default is "+ Frames_Folder
    )
    parser.add_argument(
        "--camera_name", type=str, default= None,
        help="the name of the camera (location), should be match to the name of the location that the frames were taken" + Frames_Folder
    )


    FLAGS = parser.parse_args()
    train_path = FLAGS.Frames_Folder
    name_of_video=FLAGS.camera_name
    annotation_csv_file = os.path.join(train_path, 'Annotations-export.csv')
    annotation_txt_file_temp = os.path.join(train_path, 'data_train_temp.txt')
    annotation_txt_file=os.path.join(train_path, 'data_train.txt')
    multi_df = pd.read_csv(annotation_csv_file)
    labels = multi_df['label'].unique()
    labeldict = dict(zip(labels,range(len(labels))))
    multi_df.drop_duplicates(subset=None, keep='first', inplace=True)
    convert_vott_csv_to_yolo(multi_df,name_of_video,labeldict,path = train_path, temp_name= annotation_txt_file_temp, target_name=annotation_txt_file)


