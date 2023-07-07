import open3d as o3d
from open3d import visualization
import os
import cv2
import random
import numpy as np
import json
import sys
from albumentations import (
    CLAHE, Blur, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur,
    Sharpen, Emboss, RandomBrightnessContrast, OneOf, Compose
)

n_images = int(sys.argv[2])
res = (1280, 720)
do_aug = True

object_name = sys.argv[1]
model_root = f'model_gen_{sys.argv[1]}/finetune/mesh/'
model_path = model_root+'mesh.obj'
model_img = model_root+'albedo.png'

# binary mask to polygon function
def polygonFromMask(maskedArr):
  # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
  contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
  segmentation = []
  valid_poly = 0
  for contour in contours:
  # Valid polygons have >= 6 coordinates (3 points)
     if contour.size >= 6:
        segmentation.append(contour.astype(float).flatten().tolist())
        valid_poly += 1
  if valid_poly == 0:
     raise ValueError
  return segmentation

# create COCO annotation function
def create_coco_annotation(segmentation_image, id):

    img = cv2.cvtColor(segmentation_image, cv2.COLOR_BGR2GRAY)
    # Convert the segmentation image to a binary mask
    ret, binary_mask = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    # Generate connected components from the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return None
    
    segmentation = polygonFromMask(binary_mask)

    # assuming only 1 annotation per image
    annotation = {
        "id": id,
        "image_id": id,
        "category_id": 1,
        "iscrowd": 0,
        "area": float(cv2.contourArea(contours[0])),
        "bbox": list(cv2.boundingRect(contours[0])),
        "segmentation": segmentation
    }

    # return the COCO annotation
    return annotation

# post process augmentations
aug = Compose([
        GaussNoise(p=0.2),
        OneOf([
            MotionBlur(p=2),
            MedianBlur(blur_limit=3, p=1),
            Blur(blur_limit=3, p=1),
        ], p=0.5),
        OneOf([
            CLAHE(clip_limit=2, p=1),
            Sharpen(p=1),
            Emboss(p=1),
            RandomBrightnessContrast(p=1),
        ], p=0.5),
        HueSaturationValue(p=0.2),
    ], p=0.5)

# load model and modified model for segmentation
if os.path.exists(model_path):
    mesh = o3d.io.read_triangle_mesh(model_path, True)
    size = mesh.get_max_bound() - mesh.get_min_bound()
    center = [0, 0, 0]
    scale = np.min(1 / size)
    mesh.scale(scale, center)
    print(scale)
    seg_mesh = o3d.io.read_triangle_mesh(model_path)
    seg_mesh.scale(scale, center)
    seg_mesh.paint_uniform_color([1, 0, 0])
else:
    print('model does not exist')

# get backgrounds
backgrounds = os.listdir('backgrounds')

# base coco annotation
coco_annotation = {
    "info": [],
    "licenses": [],
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": object_name,
        "supercategory": "object"
    }]
}

output_path = f'dataset_{sys.argv[1]}'

if not os.path.exists(output_path + '/'):
    os.mkdir(output_path)
    os.mkdir(output_path + '/images')
    os.mkdir(output_path + '/segmentation')
    os.mkdir(output_path + '/annotations')

# init app
vis = visualization.rendering.OffscreenRenderer(res[0], res[1])
vis.scene.show_skybox(False)
vis.scene.scene.enable_indirect_light(False)

# generate n images
for i in range(n_images):
    ############ RENDER IMAGE ############
    # set random lighting
    colour = [random.uniform(0.6, 1), random.uniform(0.6, 1), random.uniform(0.6, 1)]
    direction = [random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(-2, 2)]
    intensity = random.uniform(5000, 500000)
    vis.scene.scene.set_sun_light(direction, colour, intensity)
    vis.scene.set_lighting(vis.scene.MED_SHADOWS, [0.5, 1, 0.5])

    # set random camera
    fov = random.uniform(50, 120)
    center = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
    eye = [-x + random.uniform(-2, 2) for x in direction]
    up = [0.0, 1.0, 0.0]
    vis.setup_camera(fov, center, [-x for x in direction], up)
    
    # set random background
    background = cv2.imread('backgrounds/' + backgrounds[random.randint(0, len(backgrounds) - 1)])
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background = cv2.resize(background, (res[0], res[1]))
    background = o3d.geometry.Image(np.asarray(background))
    vis.scene.set_background([0,0,0,0],background)
    
    mat = visualization.rendering.MaterialRecord()
    albedo = o3d.io.read_image(model_img)
    mat.albedo_img = albedo
    mat.shader = "defaultLit"
    
    # add geometry
    vis.scene.add_geometry('object', mesh, mat)
    
    # render and save image
    img = vis.render_to_image()
    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_BGR2RGB)
    if do_aug:
        aug_img = aug(image=img)['image']
    else:
        aug_img = img
    cv2.imwrite(f'{output_path}/images/' + str(i) + '.png', aug_img)
    
    vis.scene.remove_geometry("object")
    
    ########## RENDER SEGMENTATION ############
    vis.scene.add_geometry("seg", seg_mesh, visualization.rendering.MaterialRecord())
    vis.scene.set_background([0,0,0,0],None)
    
    seg = vis.render_to_image()
    o3d.io.write_image(f'{output_path}/segmentation/' + str(i) + '.png', seg)
    
    vis.scene.remove_geometry("seg")
    
    ########### ADD COCO ANNOTATION ############
    annotation = create_coco_annotation(np.array(seg), i)
    if annotation:
        coco_annotation['annotations'].append(annotation)
    coco_annotation['images'].append({
        "id": i,
        "file_name": str(i) + '.png',
        "height": res[1],
        "width": res[0]
    })

# write coco annotations
with open(f'{output_path}/annotations/annotations.json', 'w') as f:
    json.dump(coco_annotation, f)

# quit app
print('Done!')
#app.quit()
#sys.exit()