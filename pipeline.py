"""Synthetic citrus grading and simulated robot sorting proof of concept."""
from pathlib import Path
import argparse, csv, json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import joblib

ROOT=Path(__file__).resolve().parents[1]
CLASSES=['grade_a','grade_b','reject']

def render(seed, label, size=128):
    rng=np.random.default_rng(seed)
    img=Image.new('RGB',(size,size),(int(rng.integers(35,60)),)*3)
    d=ImageDraw.Draw(img)
    # Conveyor-style background and illumination variation
    for y in range(0,size,14):
        shade=int(rng.integers(40,72));d.line((0,y,size,y),fill=(shade,shade,shade),width=2)
    cx=int(rng.integers(56,73));cy=int(rng.integers(56,73));r=int(rng.integers(36,45))
    base=(int(rng.integers(210,250)),int(rng.integers(110,175)),int(rng.integers(15,45)))
    d.ellipse((cx-r,cy-r,cx+r,cy+r),fill=base,outline=(130,72,18),width=2)
    for _ in range(170):
        x=int(rng.integers(cx-r,cx+r+1));y=int(rng.integers(cy-r,cy+r+1))
        if (x-cx)**2+(y-cy)**2 < (r-3)**2:
            d.point((x,y),fill=(min(255,base[0]+int(rng.integers(-20,20))),max(0,base[1]+int(rng.integers(-22,22))),base[2]))
    # Grade labels are based on deliberately different simulated blemish severity.
    count={'grade_a':0,'grade_b':3,'reject':9}[label]
    for _ in range(count):
        angle=float(rng.uniform(0,2*np.pi));rad=float(rng.uniform(0,r*.65))
        x=int(cx+rad*np.cos(angle));y=int(cy+rad*np.sin(angle));br=int(rng.integers(3,6) if label=='grade_b' else rng.integers(5,10))
        d.ellipse((x-br,y-br,x+br,y+br),fill=(int(rng.integers(65,110)),int(rng.integers(45,75)),int(rng.integers(12,35))))
    d.ellipse((cx-22,cy-26,cx-5,cy-13),fill=(255,210,110))
    img=img.filter(ImageFilter.GaussianBlur(radius=float(rng.uniform(.1,.7))))
    return img

def features(img):
    return hog(np.asarray(img.convert('L').resize((96,96))),orientations=9,pixels_per_cell=(8,8),cells_per_block=(2,2),feature_vector=True)

def command(label, confidence, frame_id):
    # Illustrative only: no Fanuc controller connection, no real robot motion.
    return {'frame_id':frame_id,'predicted_grade':label,'confidence':round(float(confidence),4),
            'robot_mode':'SIMULATION_ONLY','action':'PICK_AND_PLACE','target_bin':{'grade_a':'BIN_A','grade_b':'BIN_B','reject':'REJECT_BIN'}[label],
            'requires_hardware_safety_interlocks':True}

def main(n,seed):
    out=ROOT/'outputs';out.mkdir(exist_ok=True)
    rng=np.random.default_rng(seed)
    labels=np.repeat(CLASSES,n)
    seeds=rng.integers(0,2**31-1,size=len(labels))
    X=np.stack([features(render(int(s),str(y))) for s,y in zip(seeds,labels)])
    train_idx,test_idx=train_test_split(np.arange(len(labels)),test_size=.25,random_state=seed,stratify=labels)
    model=RandomForestClassifier(n_estimators=160,min_samples_leaf=2,random_state=seed,n_jobs=-1)
    model.fit(X[train_idx],labels[train_idx]);pred=model.predict(X[test_idx]);probs=model.predict_proba(X[test_idx])
    report=classification_report(labels[test_idx],pred,labels=CLASSES,output_dict=True,zero_division=0)
    result={'dataset':'procedurally generated illustrative citrus images','samples_total':len(labels),'train_samples':len(train_idx),'test_samples':len(test_idx),
            'test_accuracy':accuracy_score(labels[test_idx],pred),'classes':CLASSES,'confusion_matrix':confusion_matrix(labels[test_idx],pred,labels=CLASSES).tolist(),
            'classification_report':report,'limitations':['synthetic-only evaluation','no DEM','no photorealistic renderer','no real camera or fruit validation','no robot controller integration']}
    (out/'metrics.json').write_text(json.dumps(result,indent=2))
    joblib.dump(model,out/'model.joblib')
    with (out/'robot_commands.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(command('grade_a',1,0)));writer.writeheader()
        for k,i in enumerate(test_idx[:30]):writer.writerow(command(str(pred[k]),float(max(probs[k])),int(i)))
    # Contact sheet examples with ground-truth labels
    sheet=Image.new('RGB',(4*160,3*160),'white');draw=ImageDraw.Draw(sheet)
    for k in range(12):
        lab=CLASSES[k%3];im=render(int(seeds[k]),lab).resize((128,128))
        x=(k%4)*160+16;y=(k//4)*160+5;sheet.paste(im,(x,y));draw.text((x,y+130),lab,fill='black')
    sheet.save(out/'synthetic_samples.png')
    print(f"Synthetic-only held-out accuracy: {result['test_accuracy']:.3f} ({len(test_idx)} samples)")
    print('Outputs:',out)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--samples-per-class',type=int,default=90);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();main(args.samples_per_class,args.seed)
