"""
Human Detection + Filter + Face Effects Webcam App
====================================================
Uses YOLOv8 for person segmentation + OpenCV Haar for face detection.
Now with a CLICKABLE LEFT-SIDE PANEL — no keyboard needed!

Install:
    pip install ultralytics opencv-python numpy

Controls (keyboard still works too):
  --- BODY FILTERS ---         --- FACE / CHARACTER EFFECTS ---
  1  Raw                       E  Elf Ears
  2  Neon Outline              V  Vampire Teeth
  3  Cartoon                   A  Angel Wings + Halo
  4  Anime                     D  Demon Horns
  5  Pencil Sketch             M  Mermaid Look
  6  Cyberpunk                 F  Fairy Sparkles
  7  Pixel Art                 Y  Baby Face
  8  Oil Painting              C  Child Look
  9  Heat Vision               T  Teen Look
  0  Glitch                    O  Old Age
                               P  Age Progression (cycles young->old)

  B  Cycle background    K  Toggle bounding boxes    S  Save    Q  Quit

LEFT PANEL CLICK AREAS:
  Body Filters section  → click any filter to activate
  Face Effects section  → click any effect to toggle on/off
  Backgrounds section   → click any background to switch
  Action row (bottom)   → Boxes / Save / Quit buttons
"""

import cv2
import numpy as np
import time
import os
import math
import warnings
warnings.filterwarnings("ignore")
os.environ["YOLO_VERBOSE"] = "False"

from ultralytics import YOLO

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

PANEL_W        = 210          # width of the left click panel
BTN_H          = 30           # height of each button row
SECTION_H      = 22           # height of a section header
PAD            = 6            # horizontal padding inside button

# Colours (BGR)
C_BG           = (18,  18,  28)
C_SECTION_BG   = (30,  30,  48)
C_BTN          = (38,  38,  58)
C_BTN_ACTIVE   = (0,  200, 120)
C_BTN_HOVER    = (55,  55,  85)
C_BORDER       = (60,  60,  90)
C_ACTIVE_TEXT  = (10,  10,  10)
C_TEXT         = (190, 190, 210)
C_SECTION_TEXT = (0,  220, 180)
C_ACCENT       = (0,  180, 255)
C_SAVE_BTN     = (0,  140, 220)
C_QUIT_BTN     = (40,  40, 200)
C_BOX_BTN      = (160, 100,  20)

FONT           = cv2.FONT_HERSHEY_SIMPLEX
FONT_SMALL     = 0.38
FONT_SEC       = 0.40

# ─────────────────────────────────────────────────────────────────────────────
# CAMERA FINDER
# ─────────────────────────────────────────────────────────────────────────────

def find_camera(max_index=5):
    print("Scanning for cameras...")
    found = []
    for i in range(max_index + 1):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(i)
                    print(f"  [OK] Camera at index {i}")
            cap.release()
        except Exception:
            pass
    return found

def open_camera(index):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    return cap

# ─────────────────────────────────────────────────────────────────────────────
# YOLO SEGMENTOR
# ─────────────────────────────────────────────────────────────────────────────

class YOLOSegmentor:
    def __init__(self):
        print("Loading YOLOv8 segmentation model...")
        self.model  = YOLO("yolov8n-seg.pt")
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        print("Model loaded!")

    def get_mask_and_boxes(self, frame):
        h, w  = frame.shape[:2]
        mask  = np.zeros((h, w), dtype=np.uint8)
        boxes = []
        results = self.model(frame, classes=[0], verbose=False)
        r = results[0]
        if r.masks is not None:
            for seg in r.masks.data:
                m = seg.cpu().numpy()
                m = cv2.resize(m, (w, h))
                m = (m > 0.5).astype(np.uint8) * 255
                mask = cv2.bitwise_or(mask, m)
        if r.boxes is not None:
            for box in r.boxes.xyxy.cpu().numpy():
                boxes.append(box.astype(int))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        return mask, boxes

# ─────────────────────────────────────────────────────────────────────────────
# FACE DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_path     = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        self.eye_cascade  = cv2.CascadeClassifier(eye_path)

    def detect(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        return faces if len(faces) > 0 else []

    def detect_eyes(self, frame, face_rect):
        x, y, w, h = face_rect
        roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
        eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
        result = []
        for (ex, ey, ew, eh) in eyes:
            result.append((x+ex, y+ey, ew, eh))
        return result

# ─────────────────────────────────────────────────────────────────────────────
# DRAWING UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def blend_overlay(canvas, overlay_bgr, overlay_alpha, x, y):
    oh, ow = overlay_bgr.shape[:2]
    ch, cw = canvas.shape[:2]
    x1c = max(x, 0);        y1c = max(y, 0)
    x2c = min(x + ow, cw);  y2c = min(y + oh, ch)
    x1o = x1c - x;          y1o = y1c - y
    x2o = x1o + (x2c - x1c)
    y2o = y1o + (y2c - y1c)
    if x2c <= x1c or y2c <= y1c:
        return canvas
    roi = canvas[y1c:y2c, x1c:x2c].astype(np.float32)
    src = overlay_bgr[y1o:y2o, x1o:x2o].astype(np.float32)
    a   = overlay_alpha[y1o:y2o, x1o:x2o].astype(np.float32) / 255.0
    a3  = np.stack([a, a, a], axis=2)
    canvas[y1c:y2c, x1c:x2c] = np.clip(src*a3 + roi*(1-a3), 0, 255).astype(np.uint8)
    return canvas

def draw_filled_ellipse(img, center, axes, angle, color, alpha=255):
    overlay = np.zeros_like(img)
    cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1)
    a    = alpha / 255.0
    mask = (overlay.sum(axis=2) > 0).astype(np.float32) * a
    m3   = np.stack([mask, mask, mask], axis=2)
    return np.clip(img.astype(np.float32)*(1-m3) + overlay.astype(np.float32)*m3, 0, 255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# ★  FACE / CHARACTER EFFECT FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def effect_elf_ears(frame, faces, t):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        ear_w = fw // 5;  ear_h = fw // 3
        lx    = fx - ear_w // 2
        ly    = fy - ear_h + ear_h // 6
        pts_l = np.array([[lx, ly+ear_h],[lx+ear_w//2,ly],[lx+ear_w,ly+ear_h]], np.int32)
        cv2.fillPoly(out,[pts_l],(130,190,235)); cv2.polylines(out,[pts_l],True,(60,110,160),2)
        inner_l = pts_l.copy(); inner_l[:,0]+=ear_w//5; inner_l[:,1]+=ear_h//5
        cv2.polylines(out,[inner_l],False,(100,150,200),1)
        rx    = fx + fw - ear_w//2
        pts_r = np.array([[rx,ly+ear_h],[rx+ear_w//2,ly],[rx+ear_w,ly+ear_h]],np.int32)
        cv2.fillPoly(out,[pts_r],(130,190,235)); cv2.polylines(out,[pts_r],True,(60,110,160),2)
        inner_r = pts_r.copy(); inner_r[:,0]-=ear_w//5; inner_r[:,1]+=ear_h//5
        cv2.polylines(out,[inner_r],False,(100,150,200),1)
    return out

def effect_vampire_teeth(frame, faces, t):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        mx=fx+fw//2; my=fy+int(fh*0.75); tw=fw//10; th=fw//8; gap=fw//16
        for side in [-1,1]:
            tx=mx+side*gap-tw//2; ty=my
            pts=np.array([[tx,ty],[tx+tw,ty],[tx+tw//2+2,ty+th],[tx+tw//2-2,ty+th]],np.int32)
            cv2.fillPoly(out,[pts],(245,245,255)); cv2.polylines(out,[pts],True,(180,180,200),1)
            drip=int(4+4*abs(math.sin(t*1.5)))
            cv2.line(out,(tx+tw//2,ty+th),(tx+tw//2,ty+th+drip),(0,0,200),2)
            cv2.circle(out,(tx+tw//2,ty+th+drip),3,(0,0,180),-1)
    return out

def effect_angel(frame, faces, t):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        cx=fx+fw//2
        halo_y=fy-fw//6; halo_rx=fw//3; halo_ry=fw//10
        pulse=1.0+0.04*math.sin(t*3); halo_rx_p=int(halo_rx*pulse)
        for thickness,alpha in [(20,40),(12,80),(6,160),(3,255)]:
            ov=out.copy(); cv2.ellipse(ov,(cx,halo_y),(halo_rx_p,halo_ry),0,0,360,(30,200,255),thickness)
            out=cv2.addWeighted(out,1-alpha/255,ov,alpha/255,0)
        wing_w=fw*2; wing_h=int(fh*1.6); wing_y=fy+fh//3
        for side in [-1,1]:
            pts_outer=[]
            for ang in range(0,181,8):
                rad=math.radians(ang)
                ex=cx+side*int(wing_w*math.sin(rad)); ey=wing_y-int(wing_h*0.5*math.sin(rad*0.5))
                pts_outer.append([ex,ey])
            pts_outer=np.array(pts_outer,np.int32)
            ov=out.copy(); cv2.fillPoly(ov,[pts_outer],(240,240,255))
            out=cv2.addWeighted(out,0.45,ov,0.55,0)
            for i in range(10):
                r=i/10; fx2=int(cx+side*wing_w*r)
                fy2=int(wing_y-wing_h*0.4*math.sin(math.pi*r)); fy3=int(wing_y+fh*0.1)
                cv2.line(out,(fx2,fy3),(fx2,fy2),(200,200,230),1)
    return out

def effect_demon_horns(frame, faces, t):
    out = frame.copy()
    for (fx, fy, fw, fh) in faces:
        cx=fx+fw//2; horn_h=int(fw*0.5); horn_w=fw//7
        for side in [-1,1]:
            hx=cx+side*fw//4; hy=fy; sway=int(4*math.sin(t*2))
            pts=np.array([[hx-horn_w//2,hy],[hx+side*sway-2,hy-horn_h],[hx+horn_w//2,hy]],np.int32)
            shadow=pts.copy(); shadow[:,0]+=3; shadow[:,1]+=3
            cv2.fillPoly(out,[shadow],(0,0,30)); cv2.fillPoly(out,[pts],(20,20,160))
            mid=np.array([[hx+side*2,hy-5],[hx+side*sway+side*2,hy-horn_h+8],[hx+side*2+horn_w//4,hy-5]],np.int32)
            cv2.fillPoly(out,[mid],(60,60,200)); cv2.polylines(out,[pts],True,(0,0,80),2)
        ley=fy+int(fh*0.38); lex=fx+int(fw*0.28); rex=fx+int(fw*0.72); glow_r=int(fw*0.07)
        for eye_cx,eye_cy in [(lex,ley),(rex,ley)]:
            ov=out.copy(); cv2.circle(ov,(eye_cx,eye_cy),glow_r+6,(0,0,255),-1)
            out=cv2.addWeighted(out,0.6,ov,0.4,0)
            cv2.circle(out,(eye_cx,eye_cy),glow_r,(0,80,255),-1)
            cv2.circle(out,(eye_cx,eye_cy),glow_r//2,(100,200,255),-1)
    return out

def effect_mermaid(frame, faces, t):
    out=frame.copy(); h_f,w_f=frame.shape[:2]
    tint=np.zeros_like(out,dtype=np.float32); tint[:,:,0]=60; tint[:,:,1]=40
    out=np.clip(out.astype(np.float32)*0.80+tint,0,255).astype(np.uint8)
    for i in range(8):
        lx=int(w_f*(0.1+0.1*i+0.04*math.sin(t*1.2+i))); ly=int(h_f*(0.2+0.06*math.sin(t*0.8+i*1.3)))
        cv2.ellipse(out,(lx,ly),(int(30+10*math.sin(t+i)),8),30+i*10,0,360,(200,230,220),-1)
        ov2=out.copy(); cv2.ellipse(ov2,(lx,ly),(int(30+10*math.sin(t+i)),8),30+i*10,0,360,(200,230,220),-1)
        out=cv2.addWeighted(out,0.85,ov2,0.15,0)
    for (fx,fy,fw,fh) in faces:
        cx=fx+fw//2; scale_r=fw//14
        for row in range(8):
            for col in range(12):
                sx=fx+col*scale_r*2-scale_r; sy=fy+row*scale_r+(scale_r if col%2 else 0)
                if fx<=sx<=fx+fw and fy<=sy<=fy+fh:
                    hue=int((row*30+col*15+t*40)%180)
                    col_hsv=np.uint8([[[hue,200,200]]]); col_bgr=cv2.cvtColor(col_hsv,cv2.COLOR_HSV2BGR)[0][0]
                    ov3=out.copy()
                    cv2.ellipse(ov3,(sx,sy),(scale_r,scale_r-1),0,0,360,(int(col_bgr[0]),int(col_bgr[1]),int(col_bgr[2])),-1)
                    out=cv2.addWeighted(out,0.75,ov3,0.25,0)
        fin_pts=np.array([[cx-fw//3,fy],[cx-fw//5,fy-fw//5],[cx,fy-fw//4],[cx+fw//5,fy-fw//5],[cx+fw//3,fy]],np.int32)
        cv2.fillPoly(out,[fin_pts],(0,180,130)); cv2.polylines(out,[fin_pts],False,(0,220,180),2)
    return out

def effect_fairy(frame, faces, t):
    out=frame.copy(); h_f,w_f=frame.shape[:2]; rng=np.random.default_rng(7)
    positions=[]
    for i in range(40):
        positions.append((int(rng.integers(0,w_f)),int(rng.integers(0,h_f)),rng.uniform(0.5,3.0),rng.uniform(0,2*math.pi)))
    for sx,sy,sp,ph in positions:
        bri=0.5+0.5*math.sin(t*sp+ph); sz=int(1+bri*5); alpha=int(bri*255)
        col=(int(180+bri*75),int(180+bri*75),255); ov=out.copy()
        cv2.line(ov,(sx-sz,sy),(sx+sz,sy),col,1); cv2.line(ov,(sx,sy-sz),(sx,sy+sz),col,1)
        cv2.line(ov,(sx-sz//2,sy-sz//2),(sx+sz//2,sy+sz//2),col,1)
        cv2.line(ov,(sx+sz//2,sy-sz//2),(sx-sz//2,sy+sz//2),col,1)
        cv2.circle(ov,(sx,sy),max(1,sz//2),(255,255,255),-1)
        out=cv2.addWeighted(out,1-alpha/400,ov,alpha/400,0)
    for (fx,fy,fw,fh) in faces:
        for i in range(20):
            ang=t*90+i*18; rad=math.radians(ang); r=fw*0.55
            px=int(fx+fw//2+r*math.cos(rad)); py=int(fy+fh//2+r*math.sin(rad)*0.6)
            bri2=0.5+0.5*math.sin(t*3+i); sz2=int(2+bri2*4)
            cv2.circle(out,(px,py),sz2,(int(200*bri2),int(160*bri2),255),-1)
    vig=np.zeros_like(out,dtype=np.float32); vig[:,:,0]=80; vig[:,:,2]=40
    return np.clip(out.astype(np.float32)*0.92+vig*0.08,0,255).astype(np.uint8)

def effect_baby_face(frame, faces, t):
    out=frame.copy()
    for (fx,fy,fw,fh) in faces:
        cx=fx+fw//2; cy=fy+fh//2; roi=out[fy:fy+fh,fx:fx+fw]
        if roi.size==0: continue
        big_w=int(fw*1.25); big_h=int(fh*1.15); big=cv2.resize(roi,(big_w,big_h))
        bx=cx-big_w//2; by=cy-big_h//2; bx2,by2=bx+big_w,by+big_h
        bx=max(0,bx); by=max(0,by); bx2=min(out.shape[1],bx2); by2=min(out.shape[0],by2)
        cw=bx2-bx; ch=by2-by; resized_crop=cv2.resize(big,(cw,ch))
        paste_mask=np.zeros((ch,cw),dtype=np.uint8)
        cv2.ellipse(paste_mask,(cw//2,ch//2),(cw//2-2,ch//2-2),0,0,360,255,-1)
        paste_mask=cv2.GaussianBlur(paste_mask,(21,21),0); a3=np.stack([paste_mask/255.]*3,axis=2)
        out[by:by2,bx:bx2]=np.clip(resized_crop*a3+out[by:by2,bx:bx2]*(1-a3),0,255).astype(np.uint8)
        cheek_r=fw//6
        for side in [-1,1]:
            chx=cx+side*int(fw*0.32); chy=fy+int(fh*0.62); ov=out.copy()
            cv2.circle(ov,(chx,chy),cheek_r,(100,130,220),-1); out=cv2.addWeighted(out,0.75,ov,0.25,0)
        ley=fy+int(fh*0.38); lex=fx+int(fw*0.28); rex=fx+int(fw*0.72); er=fw//10
        for ex_,ey_ in [(lex,ley),(rex,ley)]:
            cv2.circle(out,(ex_,ey_),int(er*1.3),(255,255,255),-1); cv2.circle(out,(ex_,ey_),er,(50,50,50),-1)
            cv2.circle(out,(ex_+er//3,ey_-er//3),er//3,(255,255,255),-1)
        hsv=cv2.cvtColor(out[fy:fy+fh,fx:fx+fw],cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.7,0,255); hsv[:,:,2]=np.clip(hsv[:,:,2]*1.1,0,255)
        out[fy:fy+fh,fx:fx+fw]=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
    return out

def effect_child(frame, faces, t):
    out=frame.copy()
    for (fx,fy,fw,fh) in faces:
        cx=fx+fw//2; cy=fy+fh//2; roi=out[fy:fy+fh,fx:fx+fw]
        if roi.size==0: continue
        smooth=cv2.bilateralFilter(roi,9,60,60)
        hsv=cv2.cvtColor(smooth,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.80,0,255); hsv[:,:,2]=np.clip(hsv[:,:,2]*1.15,0,255)
        out[fy:fy+fh,fx:fx+fw]=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        rng=np.random.default_rng(3)
        for _ in range(18):
            frx=int(fx+rng.integers(fw//4,3*fw//4)); fry=int(fy+rng.integers(fh//3,2*fh//3))
            cv2.circle(out,(frx,fry),2,(80,100,160),-1)
        nx=cx; ny=fy+int(fh*0.58); ov=out.copy()
        cv2.circle(ov,(nx,ny),fw//14,(210,220,240),-1); out=cv2.addWeighted(out,0.80,ov,0.20,0)
        cheek_r=fw//7
        for side in [-1,1]:
            chx=cx+side*int(fw*0.30); chy=fy+int(fh*0.60); ov2=out.copy()
            cv2.circle(ov2,(chx,chy),cheek_r,(130,160,230),-1); out=cv2.addWeighted(out,0.80,ov2,0.20,0)
    return out

def effect_teen(frame, faces, t):
    out=frame.copy()
    for (fx,fy,fw,fh) in faces:
        roi=out[fy:fy+fh,fx:fx+fw]
        if roi.size==0: continue
        kernel=np.array([[0,-0.3,0],[-0.3,2.2,-0.3],[0,-0.3,0]]); sharpened=cv2.filter2D(roi,-1,kernel)
        sharpened=sharpened.astype(np.float32)
        sharpened[:,:,0]=np.clip(sharpened[:,:,0]*1.1,0,255)
        sharpened[:,:,1]=np.clip(sharpened[:,:,1]*1.05,0,255)
        sharpened[:,:,2]=np.clip(sharpened[:,:,2]*0.93,0,255)
        out[fy:fy+fh,fx:fx+fw]=sharpened.astype(np.uint8); cx=fx+fw//2
        for side in [-1,1]:
            ex=fx+(fw//10 if side==-1 else 9*fw//10); ey=fy+int(fh*0.55)
            cv2.circle(out,(ex,ey),fw//18,(40,160,200),2)
            cv2.circle(out,(ex,ey+fw//18+3),3,(40,160,200),-1)
        for side in [-1,1]:
            ex2=cx+side*fw//4; ey2=fy+int(fh*0.36)
            cv2.line(out,(ex2-fw//8,ey2),(ex2+fw//8+side*4,ey2-2),(20,20,50),2)
    return out

def effect_old_age(frame, faces, t):
    out=frame.copy()
    for (fx,fy,fw,fh) in faces:
        roi=out[fy:fy+fh,fx:fx+fw].copy()
        if roi.size==0: continue
        cx=fw//2; cy=fh//2
        hsv=cv2.cvtColor(roi,cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,1]=np.clip(hsv[:,:,1]*0.40,0,255); hsv[:,:,2]=np.clip(hsv[:,:,2]*0.90,0,255)
        roi=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
        noise=np.random.normal(0,6,roi.shape).astype(np.int16)
        roi=np.clip(roi.astype(np.int16)+noise,0,255).astype(np.uint8)
        for i in range(4):
            wy=int(fh*(0.15+i*0.05)); wave=[int(4*math.sin(x*0.3+i)) for x in range(fw)]
            pts=[(x,wy+wave[x]) for x in range(fw)]
            for j in range(len(pts)-1): cv2.line(roi,pts[j],pts[j+1],(100,100,110),1)
        for side in [-1,1]:
            ex=cx+side*fw//4; ey=int(fh*0.43)
            cv2.ellipse(roi,(ex,ey+4),(fw//10,fw//20),0,0,180,(80,85,95),2)
        for side in [-1,1]:
            sx=cx+side*fw//5
            cv2.line(roi,(sx,int(fh*0.52)),(sx+side*3,int(fh*0.72)),(90,90,100),1)
        for side in [-1,1]:
            tx=fw//8 if side==-1 else 7*fw//8
            for dy in range(int(fh*0.08)):
                roi[dy,tx-fw//16:tx+fw//16]=np.clip(roi[dy,tx-fw//16:tx+fw//16].astype(np.float32)*0.5+160,0,255).astype(np.uint8)
        out[fy:fy+fh,fx:fx+fw]=roi
    return out

AGE_PROGRESSION_STAGE=[0]

def effect_age_progression(frame, faces, t):
    stage=AGE_PROGRESSION_STAGE[0]
    funcs=[effect_baby_face,effect_child,effect_teen,f_raw,effect_old_age]
    names=["Baby","Child","Teen","Adult","Elderly"]
    out=funcs[stage](frame,faces,t)
    h_f,w_f=frame.shape[:2]
    label=f"Age Stage: {names[stage]}  (click P to advance)"
    cv2.rectangle(out,(0,h_f-55),(len(label)*11+20,h_f-32),(0,0,0),-1)
    cv2.putText(out,label,(10,h_f-38),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,230,180),1)
    return out

FACE_EFFECTS={
    ord('e'):("Elf Ears",       effect_elf_ears),
    ord('v'):("Vampire Teeth",  effect_vampire_teeth),
    ord('a'):("Angel Wings",    effect_angel),
    ord('d'):("Demon Horns",    effect_demon_horns),
    ord('m'):("Mermaid",        effect_mermaid),
    ord('f'):("Fairy Sparkles", effect_fairy),
    ord('y'):("Baby Face",      effect_baby_face),
    ord('c'):("Child Look",     effect_child),
    ord('t'):("Teen Look",      effect_teen),
    ord('o'):("Old Age",        effect_old_age),
    ord('p'):("Age Progress",   effect_age_progression),
}

# ─────────────────────────────────────────────────────────────────────────────
# BODY FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def m3f(mask):
    return cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR).astype(np.float32)/255.0

def f_raw(frame,mask): return frame.copy()

def f_neon_outline(frame,mask):
    dark=  (frame*0.12).astype(np.uint8)
    hard=  (mask>127).astype(np.uint8)*255
    outline=cv2.Canny(hard,30,100)
    outline=cv2.dilate(outline,np.ones((3,3),np.uint8),iterations=3)
    gw=cv2.GaussianBlur(outline,(25,25),0); gn=cv2.GaussianBlur(outline,(7,7),0)
    neon=np.zeros_like(frame,dtype=np.float32)
    neon[:,:,0]+=gw.astype(np.float32)*1.2; neon[:,:,1]+=gw.astype(np.float32)*1.4; neon[:,:,2]+=gw.astype(np.float32)*0.3
    neon[:,:,0]+=gn.astype(np.float32)*1.5; neon[:,:,1]+=gn.astype(np.float32)*1.5; neon[:,:,2]+=gn.astype(np.float32)*1.5
    neon=np.clip(neon,0,255).astype(np.uint8); alpha=m3f(mask)
    body=(frame*alpha*0.35+dark*(1-alpha)).astype(np.uint8)
    return cv2.add(body,neon)

def f_cartoon(frame,mask):
    data=frame.reshape((-1,3)).astype(np.float32); crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,8,1.0)
    _,labels,centers=cv2.kmeans(data,8,None,crit,3,cv2.KMEANS_RANDOM_CENTERS)
    quant=np.uint8(centers)[labels.flatten()].reshape(frame.shape)
    gray=cv2.medianBlur(cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY),7)
    edges=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,6)
    cartoon=cv2.bitwise_and(quant,cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR))
    hsv=cv2.cvtColor(cartoon,cv2.COLOR_BGR2HSV).astype(np.float32); hsv[:,:,1]=np.clip(hsv[:,:,1]*1.8,0,255)
    return cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)

def f_anime(frame,mask):
    s=frame.copy()
    for _ in range(5): s=cv2.bilateralFilter(s,9,75,75)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges=cv2.adaptiveThreshold(cv2.medianBlur(gray,5),255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,9,3)
    anime=cv2.bitwise_and(s,cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR))
    hsv=cv2.cvtColor(anime,cv2.COLOR_BGR2HSV).astype(np.float32); hsv[:,:,1]=np.clip(hsv[:,:,1]*2.0,0,255)
    result=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
    bright=cv2.cvtColor(result,cv2.COLOR_BGR2GRAY); _,hi=cv2.threshold(bright,200,255,cv2.THRESH_BINARY)
    glow=cv2.GaussianBlur(cv2.cvtColor(hi,cv2.COLOR_GRAY2BGR),(15,15),0)
    return cv2.addWeighted(result,1.0,glow,0.3,0)

def f_pencil_sketch(frame,mask):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); blur=cv2.GaussianBlur(cv2.bitwise_not(gray),(25,25),0)
    sketch=cv2.divide(gray,cv2.bitwise_not(blur),scale=256.0)
    noise=np.random.normal(0,4,sketch.shape).astype(np.int16)
    sketch=np.clip(sketch.astype(np.int16)+noise,0,255).astype(np.uint8)
    return cv2.cvtColor(sketch,cv2.COLOR_GRAY2BGR)

def f_cyberpunk(frame,mask):
    dark=(frame*0.08).astype(np.uint8); gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    edges=cv2.dilate(cv2.Canny(gray,30,100),np.ones((2,2),np.uint8)); eb=cv2.GaussianBlur(edges,(5,5),0)
    neon=np.zeros_like(frame,dtype=np.float32)
    neon[:,:,0]=np.roll(eb,5,axis=1).astype(np.float32)*2.5
    neon[:,:,1]=np.roll(eb,-5,axis=0).astype(np.float32)*1.5
    neon[:,:,2]=eb.astype(np.float32)*2.0
    neon=np.clip(neon,0,255).astype(np.uint8)
    tinted=frame.copy().astype(np.float32)
    tinted[:,:,0]=np.clip(tinted[:,:,0]*1.4,0,255); tinted[:,:,2]=np.clip(tinted[:,:,2]*0.6,0,255)
    tinted=tinted.astype(np.uint8); alpha=m3f(mask)
    body=(tinted*alpha*0.6+dark*(1-alpha)).astype(np.uint8)
    return cv2.add(body,neon)

def f_pixel_art(frame,mask):
    h,w=frame.shape[:2]; px=10
    sm=cv2.resize(frame,(w//px,h//px),interpolation=cv2.INTER_LINEAR)
    pix=cv2.resize(sm,(w,h),interpolation=cv2.INTER_NEAREST)
    return (pix//(256//6))*(256//6)

def f_oil_paint(frame,mask):
    oil=frame.copy()
    for _ in range(7): oil=cv2.bilateralFilter(oil,9,150,150)
    gray=cv2.cvtColor(oil,cv2.COLOR_BGR2GRAY)
    emboss=cv2.cvtColor(cv2.filter2D(gray,-1,np.array([[-2,-1,0],[-1,1,1],[0,1,2]],dtype=np.float32)),cv2.COLOR_GRAY2BGR)
    hsv=cv2.cvtColor(oil,cv2.COLOR_BGR2HSV).astype(np.float32); hsv[:,:,1]=np.clip(hsv[:,:,1]*1.3,0,255)
    oil=cv2.cvtColor(hsv.astype(np.uint8),cv2.COLOR_HSV2BGR)
    return cv2.addWeighted(oil,0.88,emboss,0.12,0)

def f_heat_vision(frame,mask):
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY); alpha=mask.astype(np.float32)/255.0
    boosted=np.clip(gray.astype(np.float32)+alpha*60,0,255).astype(np.uint8)
    return cv2.GaussianBlur(cv2.applyColorMap(boosted,cv2.COLORMAP_INFERNO),(3,3),0)

def f_glitch(frame,mask):
    h,w=frame.shape[:2]; out=frame.copy(); shift=8
    b,g,r=cv2.split(out); out=cv2.merge([np.roll(b,-shift,axis=1),g,np.roll(r,shift,axis=1)])
    rng=np.random.default_rng(int(time.time()*10)%10000)
    for _ in range(6):
        y1=int(rng.integers(0,h)); y2=min(y1+int(rng.integers(2,12)),h)
        out[y1:y2]=np.roll(out[y1:y2],int(rng.integers(-30,30)),axis=1)
    out[::4,:,1]=np.clip(out[::4,:,1].astype(np.int16)+40,0,255).astype(np.uint8)
    return out

BODY_FILTERS={
    ord('1'):("Raw",           f_raw),
    ord('2'):("Neon Outline",  f_neon_outline),
    ord('3'):("Cartoon",       f_cartoon),
    ord('4'):("Anime",         f_anime),
    ord('5'):("Pencil Sketch", f_pencil_sketch),
    ord('6'):("Cyberpunk",     f_cyberpunk),
    ord('7'):("Pixel Art",     f_pixel_art),
    ord('8'):("Oil Painting",  f_oil_paint),
    ord('9'):("Heat Vision",   f_heat_vision),
    ord('0'):("Glitch",        f_glitch),
}

# Ordered lists for sidebar layout
BODY_FILTER_LIST=[
    (ord('1'),"1","Raw"),       (ord('2'),"2","Neon Outline"),
    (ord('3'),"3","Cartoon"),   (ord('4'),"4","Anime"),
    (ord('5'),"5","Pencil"),    (ord('6'),"6","Cyberpunk"),
    (ord('7'),"7","Pixel Art"), (ord('8'),"8","Oil Paint"),
    (ord('9'),"9","Heat Vis"),  (ord('0'),"0","Glitch"),
]

FACE_EFFECT_LIST=[
    (ord('e'),"E","Elf Ears"),      (ord('v'),"V","Vampire"),
    (ord('a'),"A","Angel Wings"),   (ord('d'),"D","Demon Horns"),
    (ord('m'),"M","Mermaid"),       (ord('f'),"F","Fairy"),
    (ord('y'),"Y","Baby Face"),     (ord('c'),"C","Child"),
    (ord('t'),"T","Teen"),          (ord('o'),"O","Old Age"),
    (ord('p'),"P","Age Prog"),
]

BG_LIST=[
    ("Space",   (80, 0, 100)),
    ("Forest",  (20,130,  20)),
    ("Ocean",   (160, 80,  0)),
    ("Sunset",  (20,100,200)),
    ("Matrix",  (0, 160,  0)),
    ("None",    (60, 60,  60)),
]

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUNDS
# ─────────────────────────────────────────────────────────────────────────────

def make_space_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        r=y/h; bg[y]=[int(30*r),0,int(60+60*r)]
    rng=np.random.default_rng(42)
    xs,ys=rng.integers(0,w,400),rng.integers(0,h,400); br=rng.integers(150,255,400)
    for x,y,b in zip(xs,ys,br): cv2.circle(bg,(int(x),int(y)),1,(int(b),int(b),int(b)),-1)
    for cx,cy,radius,col in [(w//3,h//3,90,(80,0,130)),(2*w//3,2*h//3,70,(0,50,140))]:
        nm=np.zeros((h,w),dtype=np.float32); cv2.circle(nm,(cx,cy),radius,1.0,-1)
        nm=cv2.GaussianBlur(nm,(101,101),0)
        for c,v in enumerate(col): bg[:,:,c]=np.clip(bg[:,:,c].astype(np.float32)+nm*v,0,255).astype(np.uint8)
    return bg

def make_forest_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8); bg[:h//2]=[180,220,120]; bg[h//2:]=[25,90,15]
    cv2.circle(bg,(w-80,60),50,(40,210,255),-1); rng=np.random.default_rng(5)
    for tx in range(20,w,80):
        th=int(rng.integers(70,130)); ty=h//2
        cv2.rectangle(bg,(tx+28,ty-th),(tx+42,ty+5),(35,70,15),-1)
        cv2.circle(bg,(tx+35,ty-th),55,(15,130,25),-1)
        cv2.circle(bg,(tx+18,ty-th+25),38,(20,150,30),-1)
        cv2.circle(bg,(tx+52,ty-th+25),38,(18,140,28),-1)
    return bg

def make_ocean_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        r=y/h; bg[y]=[int(160*(1-r)),int(90+60*r),int(190+65*r)]
    for rx in range(0,w,80):
        pts=np.array([[rx,0],[rx+40,0],[rx+130,h],[rx+80,h]],np.int32)
        ov=bg.copy(); cv2.fillPoly(ov,[pts],(200,230,255)); bg=cv2.addWeighted(bg,0.91,ov,0.09,0)
    rng=np.random.default_rng(7)
    for _ in range(40): cv2.circle(bg,(int(rng.integers(0,w)),int(rng.integers(0,h))),int(rng.integers(3,14)),(220,240,255),1)
    return bg

def make_sunset_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8)
    for y in range(h):
        r=y/h; bg[y]=[int(20+40*r),int(60*(1-r)+30*r),int(180*(1-r)+10*r)]
    cv2.circle(bg,(w//2,h//2+50),75,(20,170,255),-1)
    cv2.circle(bg,(w//2,h//2+50),90,(10,110,190),10)
    return bg

def make_matrix_bg(h,w):
    bg=np.zeros((h,w,3),dtype=np.uint8); rng=np.random.default_rng(99); chars="01ABXYZ"
    for col_x in range(0,w,14):
        ln=int(rng.integers(5,max(6,h//14))); sy=int(rng.integers(0,h))
        for i in range(ln):
            y=(sy+i*14)%h; b=max(40,255-i*20)
            cv2.putText(bg,chars[int(rng.integers(0,len(chars)))],(col_x,y),cv2.FONT_HERSHEY_SIMPLEX,0.4,(0,b,0),1)
    return bg

_BG_CACHE={}
BG_NAMES=["Space","Forest","Ocean","Sunset","Matrix"]
BG_FUNCS=[make_space_bg,make_forest_bg,make_ocean_bg,make_sunset_bg,make_matrix_bg]

def get_bg(name,h,w):
    key=(name,h,w)
    if key not in _BG_CACHE: _BG_CACHE[key]=BG_FUNCS[BG_NAMES.index(name)](h,w)
    return _BG_CACHE[key].copy()

def composite(person,bg,mask):
    a=cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR).astype(np.float32)/255.0
    return np.clip(person*a+bg*(1-a),0,255).astype(np.uint8)

# ─────────────────────────────────────────────────────────────────────────────
# ★  LEFT CLICK PANEL
# ─────────────────────────────────────────────────────────────────────────────

class SidebarUI:
    """
    Renders a fixed-width left panel onto the combined frame and records
    bounding boxes of every clickable button so mouse events can be routed.

    Coordinate system: the panel lives at x=[0, PANEL_W) in the FULL window
    (panel + video side by side).
    """

    def __init__(self):
        # Each entry: (y1, y2, action_type, action_value)
        self.hit_boxes: list = []

    # ── helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _rect(img, x1, y1, x2, y2, col, radius=4):
        """Filled rounded-ish rectangle (simple approximation)."""
        cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), col, -1)
        cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), col, -1)
        for cx, cy in [(x1+radius,y1+radius),(x2-radius,y1+radius),
                       (x1+radius,y2-radius),(x2-radius,y2-radius)]:
            cv2.circle(img, (cx,cy), radius, col, -1)

    @staticmethod
    def _border(img, x1, y1, x2, y2, col, thick=1):
        cv2.rectangle(img, (x1,y1), (x2,y2), col, thick)

    def _section_header(self, panel, y, label):
        cv2.rectangle(panel, (0,y), (PANEL_W, y+SECTION_H), C_SECTION_BG, -1)
        # accent line on left
        cv2.rectangle(panel, (0,y), (3, y+SECTION_H), C_BTN_ACTIVE, -1)
        cv2.putText(panel, label, (8, y+SECTION_H-6),
                    FONT, FONT_SEC, C_SECTION_TEXT, 1, cv2.LINE_AA)
        return y + SECTION_H

    def _button(self, panel, y, label, shortcut, active, col_active=None):
        """Draw one button row. Returns bottom y."""
        x1,y1,x2,y2 = PAD, y, PANEL_W-PAD, y+BTN_H-2
        bg_col = (col_active or C_BTN_ACTIVE) if active else C_BTN
        self._rect(panel, x1, y1, x2, y2, bg_col)
        self._border(panel, x1, y1, x2, y2,
                     (col_active or C_BTN_ACTIVE) if active else C_BORDER)
        txt_col = C_ACTIVE_TEXT if active else C_TEXT
        sc_col  = (20,20,20) if active else (100,100,130)
        cv2.putText(panel, f"[{shortcut}]", (x1+4, y1+BTN_H-10),
                    FONT, 0.32, sc_col, 1, cv2.LINE_AA)
        cv2.putText(panel, label, (x1+26, y1+BTN_H-10),
                    FONT, FONT_SMALL, txt_col, 1, cv2.LINE_AA)
        return y + BTN_H

    def _bg_button(self, panel, y, label, dot_col, active):
        x1,y1,x2,y2 = PAD, y, PANEL_W-PAD, y+BTN_H-2
        bg_col = C_BTN_ACTIVE if active else C_BTN
        self._rect(panel, x1, y1, x2, y2, bg_col)
        self._border(panel, x1, y1, x2, y2, C_BTN_ACTIVE if active else C_BORDER)
        # coloured dot
        dot_x = x1+12; dot_y = (y1+y2)//2
        cv2.circle(panel, (dot_x, dot_y), 6, dot_col, -1)
        cv2.circle(panel, (dot_x, dot_y), 6, (200,200,200) if active else (80,80,80), 1)
        txt_col = C_ACTIVE_TEXT if active else C_TEXT
        cv2.putText(panel, label, (x1+26, y1+BTN_H-10),
                    FONT, FONT_SMALL, txt_col, 1, cv2.LINE_AA)
        return y + BTN_H

    def _action_button(self, panel, y, label, col):
        x1,y1,x2,y2 = PAD, y, PANEL_W-PAD, y+BTN_H-2
        self._rect(panel, x1, y1, x2, y2, col)
        self._border(panel, x1, y1, x2, y2, (200,200,200))
        tw,_ = cv2.getTextSize(label, FONT, FONT_SMALL, 1)[0], 0
        tx = (PANEL_W - tw[0]) // 2
        cv2.putText(panel, label, (tx, y1+BTN_H-10),
                    FONT, FONT_SMALL, (230,230,230), 1, cv2.LINE_AA)
        return y + BTN_H

    # ── main render ───────────────────────────────────────────────────────────
    def render(self, frame_h, body_key, face_key, bg_idx, show_boxes, fps, people, age_stage):
        """
        Build a (frame_h × PANEL_W) panel image and a fresh hit_boxes list.
        Returns: panel (BGR ndarray)
        """
        panel = np.zeros((frame_h, PANEL_W, 3), dtype=np.uint8)
        panel[:] = C_BG
        # subtle scanline effect
        panel[::2, :] = np.clip(panel[::2].astype(np.int16) + 4, 0, 255).astype(np.uint8)

        self.hit_boxes = []
        y = 0

        # ── logo bar ──────────────────────────────────────────────────────────
        cv2.rectangle(panel, (0,0), (PANEL_W, 38), (12,12,24), -1)
        cv2.rectangle(panel, (0,36), (PANEL_W, 38), C_BTN_ACTIVE, -1)
        cv2.putText(panel, "FILTER CAM", (8, 26),
                    FONT, 0.50, C_ACCENT, 1, cv2.LINE_AA)
        fps_col = (0,220,80) if fps>=20 else (0,180,255) if fps>=12 else (0,60,220)
        cv2.putText(panel, f"{fps:.0f}fps  P:{people}", (PANEL_W-95, 26),
                    FONT, 0.38, fps_col, 1, cv2.LINE_AA)
        y = 40

        # ── BODY FILTERS ──────────────────────────────────────────────────────
        y = self._section_header(panel, y, "  BODY FILTER")
        for (key, sc, lbl) in BODY_FILTER_LIST:
            y0 = y
            y  = self._button(panel, y, lbl, sc, key==body_key)
            self.hit_boxes.append((y0, y, 'body', key))

        y += 3  # small gap

        # ── FACE EFFECTS ──────────────────────────────────────────────────────
        y = self._section_header(panel, y, "  FACE EFFECT")
        for (key, sc, lbl) in FACE_EFFECT_LIST:
            is_active = (key == face_key)
            display   = lbl
            if key==ord('p') and is_active:
                ages=["Baby","Child","Teen","Adult","Old"]
                display = f"Age:{ages[age_stage]}"
            y0 = y
            y  = self._button(panel, y, display, sc, is_active)
            self.hit_boxes.append((y0, y, 'face', key))

        y += 3

        # ── BACKGROUND ────────────────────────────────────────────────────────
        y = self._section_header(panel, y, "  BACKGROUND")
        for i, (name, dot_col) in enumerate(BG_LIST):
            active = (i < len(BG_NAMES) and BG_NAMES[i-1 if i>0 else 0]==name and bg_idx==i-1) \
                     or (name=="None" and bg_idx==-1) \
                     or (i < len(BG_NAMES) and bg_idx==i)
            # simpler active check
            if name=="None":
                active = (bg_idx==-1)
            else:
                active = (bg_idx == i)
            y0 = y
            y  = self._bg_button(panel, y, name, dot_col, active)
            val = i if name!="None" else -1
            self.hit_boxes.append((y0, y, 'bg', val))

        y += 4

        # ── ACTION BUTTONS ────────────────────────────────────────────────────
        y = self._section_header(panel, y, "  ACTIONS")

        # Bounding-box toggle
        box_col = (C_BOX_BTN[0]+30, C_BOX_BTN[1]+30, C_BOX_BTN[2]) if show_boxes else C_BOX_BTN
        lbl_box = "[K] Boxes ON " if show_boxes else "[K] Boxes OFF"
        y0=y; y=self._action_button(panel, y, lbl_box, box_col)
        self.hit_boxes.append((y0, y, 'boxes', None))

        # Save
        y0=y; y=self._action_button(panel, y, "[S] Save Screenshot", C_SAVE_BTN)
        self.hit_boxes.append((y0, y, 'save', None))

        # Quit
        y0=y; y=self._action_button(panel, y, "[Q] Quit", C_QUIT_BTN)
        self.hit_boxes.append((y0, y, 'quit', None))

        # ── bottom accent ─────────────────────────────────────────────────────
        cv2.rectangle(panel, (0, frame_h-2), (PANEL_W, frame_h), C_BTN_ACTIVE, -1)

        # vertical right border with glow
        for i,alpha in enumerate([0.12,0.25,0.50,1.0]):
            x = PANEL_W - 1 - i
            if x >= 0:
                panel[:,x] = np.clip(
                    panel[:,x].astype(np.float32)*(1-alpha) +
                    np.array(C_BTN_ACTIVE,dtype=np.float32)*alpha, 0, 255
                ).astype(np.uint8)

        return panel

    def hit_test(self, x, y):
        """
        Returns (action_type, action_value) if (x,y) is inside panel,
        else None.  x,y are in FULL-WINDOW coordinates.
        """
        if x < 0 or x >= PANEL_W:
            return None
        for (y1, y2, atype, aval) in self.hit_boxes:
            if y1 <= y < y2:
                return (atype, aval)
        return None

# ─────────────────────────────────────────────────────────────────────────────
# MOUSE CALLBACK STATE
# ─────────────────────────────────────────────────────────────────────────────

class MouseState:
    def __init__(self):
        self.x = 0
        self.y = 0
        self.clicked_action = None   # (type, value) set on left-button-up

def make_mouse_callback(ms: MouseState):
    def callback(event, x, y, flags, param):
        ms.x = x
        ms.y = y
        if event == cv2.EVENT_LBUTTONUP:
            ms.clicked_action = (x, y)   # store raw coords, resolved later
    return callback

# ─────────────────────────────────────────────────────────────────────────────
# HUD (overlaid on the video portion only)
# ─────────────────────────────────────────────────────────────────────────────

def draw_hud(frame, body_name, face_name, bg_name, fps, people):
    h,w=frame.shape[:2]
    ov=frame.copy(); cv2.rectangle(ov,(0,0),(w,50),(0,0,0),-1)
    frame=cv2.addWeighted(ov,0.45,frame,0.55,0)
    cv2.putText(frame,f"Body: {body_name}",(8,20),FONT,0.56,(0,255,180),1,cv2.LINE_AA)
    info=f"Face: {face_name or 'None'}   BG: {bg_name or 'None'}   People: {people}"
    cv2.putText(frame,info,(8,42),FONT,0.40,(140,140,160),1,cv2.LINE_AA)
    return frame

# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    cameras = find_camera()
    if not cameras:
        print("No webcam found!")
        input("Press Enter to exit..."); return

    cam_index = cameras[0]
    if len(cameras)>1:
        print(f"Multiple cameras: {cameras}  — Enter index: ",end="",flush=True)
        try: cam_index=cameras[int(input().strip())]
        except Exception: cam_index=cameras[0]

    try:
        seg  = YOLOSegmentor()
        face = FaceDetector()
    except Exception as e:
        print(f"Model load failed: {e}"); input("Press Enter to exit..."); return

    cap = open_camera(cam_index)
    if not cap.isOpened():
        print(f"Cannot open camera {cam_index}."); input("Press Enter..."); return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    body_filter_key = ord('1')
    face_effect_key = None
    bg_idx          = -1
    show_boxes      = False
    prev_time       = time.time()
    save_dir        = os.path.expanduser("~/Pictures")
    os.makedirs(save_dir,exist_ok=True)

    sidebar  = SidebarUI()
    ms       = MouseState()
    WIN_NAME = "Human Filter Cam"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN_NAME, make_mouse_callback(ms))

    print("\n========================================================")
    print("  Human Detection + Filter + Face Effects Cam — Ready!")
    print("  Click the LEFT PANEL or use keyboard shortcuts")
    print("  Q / Esc  to quit")
    print("========================================================\n")

    quit_flag = False

    while not quit_flag:
        ret, frame = cap.read()
        if not ret: continue
        frame = cv2.flip(frame,1)
        now   = time.time()
        t     = now

        # ── YOLO ─────────────────────────────────────────────────────────
        mask, boxes = seg.get_mask_and_boxes(frame)

        # ── Body filter ───────────────────────────────────────────────────
        b_name, b_fn = BODY_FILTERS[body_filter_key]
        filtered = b_fn(frame, mask)

        # ── Virtual background ────────────────────────────────────────────
        if bg_idx >= 0:
            h_f,w_f = frame.shape[:2]
            bg       = get_bg(BG_NAMES[bg_idx],h_f,w_f)
            filtered = composite(filtered,bg,mask)

        # ── Bounding boxes ────────────────────────────────────────────────
        if show_boxes:
            for i,box in enumerate(boxes):
                x1,y1,x2,y2=box
                cv2.rectangle(filtered,(x1,y1),(x2,y2),(0,255,200),2)
                cv2.putText(filtered,f"Person {i+1}",(x1+4,y1-6),FONT,0.52,(255,255,255),1)

        # ── Face effect ───────────────────────────────────────────────────
        f_name=None
        if face_effect_key is not None:
            faces=face.detect(frame)
            f_name,f_fn=FACE_EFFECTS[face_effect_key]
            if face_effect_key==ord('p'):
                filtered=effect_age_progression(filtered,faces,t)
            else:
                filtered=f_fn(filtered,faces,t)

        # ── HUD on video ──────────────────────────────────────────────────
        fps=1.0/max(now-prev_time,1e-6); prev_time=now
        bg_name=BG_NAMES[bg_idx] if bg_idx>=0 else None
        filtered=draw_hud(filtered,b_name,f_name,bg_name,fps,len(boxes))

        # ── Build left panel ──────────────────────────────────────────────
        h_f,w_f=filtered.shape[:2]
        panel=sidebar.render(h_f, body_filter_key, face_effect_key,
                             bg_idx, show_boxes, fps, len(boxes),
                             AGE_PROGRESSION_STAGE[0])

        # ── Composite: panel | video ──────────────────────────────────────
        display=np.hstack([panel, filtered])

        cv2.imshow(WIN_NAME, display)

        # ── Handle mouse click (resolved against sidebar hit_boxes) ───────
        if ms.clicked_action is not None:
            cx, cy = ms.clicked_action
            ms.clicked_action = None
            hit = sidebar.hit_test(cx, cy)
            if hit:
                atype, aval = hit
                if atype == 'body':
                    body_filter_key = aval
                elif atype == 'face':
                    if face_effect_key == aval:
                        if aval == ord('p'):
                            AGE_PROGRESSION_STAGE[0]=(AGE_PROGRESSION_STAGE[0]+1)%5
                        else:
                            face_effect_key=None
                    else:
                        face_effect_key=aval
                        if aval==ord('p'): AGE_PROGRESSION_STAGE[0]=0
                elif atype == 'bg':
                    bg_idx = aval
                elif atype == 'boxes':
                    show_boxes=not show_boxes
                elif atype == 'save':
                    ts=time.strftime("%Y%m%d_%H%M%S")
                    path=os.path.join(save_dir,f"filterCam_{ts}.png")
                    cv2.imwrite(path,display)
                    print(f"Saved → {path}")
                elif atype == 'quit':
                    quit_flag=True

        # ── Keyboard ──────────────────────────────────────────────────────
        key=cv2.waitKey(1)&0xFF
        if key in (ord('q'),27):
            break
        elif key in BODY_FILTERS:
            body_filter_key=key
        elif key in FACE_EFFECTS:
            if face_effect_key==key:
                if key==ord('p'): AGE_PROGRESSION_STAGE[0]=(AGE_PROGRESSION_STAGE[0]+1)%5
                else: face_effect_key=None
            else:
                face_effect_key=key
        elif key==ord('b'):
            bg_idx=(bg_idx+1)%len(BG_NAMES)
        elif key==ord('n'):
            bg_idx=-1
        elif key==ord('k'):
            show_boxes=not show_boxes
        elif key==ord('s'):
            ts=time.strftime("%Y%m%d_%H%M%S")
            path=os.path.join(save_dir,f"filterCam_{ts}.png")
            cv2.imwrite(path,display)
            print(f"Saved → {path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Bye!")

if __name__=="__main__":
    main()