import pickle
import os

mcb_root = os.path.dirname(os.path.abspath(__file__)) + '/dataset/mcb_jpg/'
all_classes = [i for i in sorted(os.listdir(mcb_root)) if i[-3:] != 'npy']

unseen_classes = ['Bushes', 'Castor', 'Fan', 'Hinge', 'Impeller', 'Knob', 'Nozzle', 'Turbine', 'Wheel']
seen_classes = [c for c in all_classes if c not in unseen_classes]

if not os.path.exists('./dataset'):
    os.mkdir('./dataset')
if not os.path.exists('./dataset/mcb'):
    os.mkdir('./dataset/mcb')
save_folder = './dataset/mcb'


def create_pickle(root, seen, seen_classes, unseen_classes, save_folder):
    data = {}
    if seen == True:
        classes = seen_classes
    else:
        classes = unseen_classes

    data = {}
    for cat in ['train', 'test']:
        class_counter = 0
        data[cat] = {}
        data[cat]['img_pth'] = []
        data[cat]['img2_pth'] = []
        data[cat]['labels'] = []
        for cls in classes:
            for img in sorted(os.listdir(root + '/' + cls + '/' + cat)):
                obj = img[:-8]
                view = int(img[-7:-4])
                data[cat]['img_pth'].append(root + '/' + cls + '/' + cat + '/' + img)
                data[cat]['labels'].append(class_counter)
                other_view_pth = []
                for v in range(1, 13):
                    if not v == view:
                        other_view_pth.append(root + '/' + cls + '/' + cat + '/' + obj + '_' + str(v).zfill(3) + '.jpg')
                data[cat]['img2_pth'].append(other_view_pth)
            class_counter += 1

    if seen == True:
        save_path = save_folder + '/seen.pickle'
    else:
        save_path = save_folder + '/unseen.pickle'

    with open(save_path, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


create_pickle(mcb_root, seen=True, seen_classes=seen_classes, unseen_classes=unseen_classes,
              save_folder=save_folder)
create_pickle(mcb_root, seen=False, seen_classes=seen_classes, unseen_classes=unseen_classes,
              save_folder=save_folder)
