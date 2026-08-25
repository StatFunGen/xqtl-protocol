from PIL import Image, ImageDraw, ImageFont
import os, matplotlib
W,H=1120,300
FD=os.path.join(os.path.dirname(matplotlib.__file__),'mpl-data','fonts','ttf')
FB=os.path.join(FD,'DejaVuSans-Bold.ttf'); FR=os.path.join(FD,'DejaVuSans.ttf')
BG=(247,250,253); CARD=(255,255,255); BORDER=(221,228,234)
BADGE_BG=(232,237,242); BADGE_TX=(138,152,165); TITLE=(46,58,69); SUB=(138,152,165)
ARROW=(184,194,203); BLUE=(26,102,179); BSOFT=(198,216,236); GREEN=(127,191,138)
GREY=(206,214,221); SCREEN=(240,245,250); IDLE=(238,242,246)
CW,CH,CY=248,226,24
GAP=(W-4*CW-60)//3
XS=[30+i*(CW+GAP) for i in range(4)]
STEPS=[('INSTALL','SoS + pixi environment'),('CLONE','xqtl-protocol repository'),
       ('TRY FIXTURES','tests/fixtures/protocol_example.*'),('RUN','sos run pipeline/....ipynb')]
ft=ImageFont.truetype(FB,15); fs=ImageFont.truetype(FR,10)
fb=ImageFont.truetype(FB,11); fy=ImageFont.truetype(FB,8)
def ctr(d,t,f,cx,y,c):
    w=d.textbbox((0,0),t,font=f)[2]; d.text((cx-w/2,y),t,font=f,fill=c)
def i1(d,cx,cy,t):
    mw,mh=96,62; x0,y0=cx-mw//2,cy-mh//2-6
    d.rounded_rectangle([x0,y0,x0+mw,y0+mh],6,fill=SCREEN,outline=GREY,width=2)
    d.rounded_rectangle([cx-8,y0+mh,cx+8,y0+mh+7],2,fill=GREY)
    d.rounded_rectangle([cx-26,y0+mh+7,cx+26,y0+mh+11],2,fill=GREY)
    for i,l in enumerate(('SoS','pixi')):
        bx=x0+12+i*44; on=t>(0.12+i*0.18)
        d.rounded_rectangle([bx,y0+12,bx+36,y0+27],4,fill=(BSOFT if on else IDLE))
        w=d.textbbox((0,0),l,font=fy)[2]
        d.text((bx+18-w/2,y0+16),l,font=fy,fill=(BLUE if on else GREY))
    bw=mw-24; bx,by=x0+12,y0+mh-18
    d.rounded_rectangle([bx,by,bx+bw,by+8],4,fill=(235,240,245))
    fw=int(bw*min(1.0,t/0.85))
    if fw>4: d.rounded_rectangle([bx,by,bx+fw,by+8],4,fill=BLUE)
def i2(d,cx,cy,t):
    fw,fh=100,66; x0,y0=cx-fw//2,cy-fh//2
    d.polygon([(x0,y0+10),(x0+38,y0+10),(x0+46,y0),(x0+fw,y0),(x0+fw,y0+fh),(x0,y0+fh)],fill=(250,252,254),outline=GREY)
    d.line([(x0,y0+10),(x0+38,y0+10),(x0+46,y0),(x0+fw,y0)],fill=GREY,width=2)
    for i in range(3):
        ap=0.15+i*0.22
        if t<ap: continue
        p=min(1.0,(t-ap)/0.16); bx=x0+14+i*28; by=y0+fh-16-int(24*p)
        d.rounded_rectangle([bx,by,bx+20,by+26],3,fill=(GREEN if i==2 else BSOFT))
def i3(d,cx,cy,t):
    s,g=20,8; gw=4*s+3*g; gh=2*s+g; x0,y0=cx-gw//2,cy-gh//2-6
    for r in range(2):
        for c in range(4):
            i=r*4+c; on=t>0.10+i*0.09
            x,y=x0+c*(s+g),y0+r*(s+g)
            d.rounded_rectangle([x,y,x+s,y+s],3,fill=(BSOFT if on else IDLE))
            if on: d.line([(x+5,y+10),(x+8,y+14),(x+15,y+6)],fill=BLUE,width=2)
    by=y0+gh+14
    d.line([(x0,by),(x0+gw,by)],fill=(232,237,242),width=3)
    d.line([(x0,by),(x0+int(gw*min(1.0,t/0.85)),by)],fill=GREEN,width=3)
def i4(d,cx,cy,t):
    sp,r=96,11; x0=cx-sp//2; ys=cy-4; st=sp//2
    d.line([(x0,ys),(x0+sp,ys)],fill=(232,237,242),width=3)
    p=min(1.0,t/0.85)
    d.line([(x0,ys),(x0+int(sp*p),ys)],fill=BLUE,width=3)
    for i in range(3):
        x=x0+i*st; on=p>=i/2-0.02
        d.ellipse([x-r,ys-r,x+r,ys+r],fill=(BLUE if on else IDLE),outline=(BLUE if on else GREY),width=2)
        if i==0: d.polygon([(x-3,ys-5),(x-3,ys+5),(x+5,ys)],fill=((255,255,255) if on else GREY))
IC=[i1,i2,i3,i4]
def frame(t):
    im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
    for i,(x,(ti,su)) in enumerate(zip(XS,STEPS)):
        d.rounded_rectangle([x,CY,x+CW,CY+CH],10,fill=CARD,outline=BORDER,width=2)
        bx,by=x+24,CY+24
        d.ellipse([bx-12,by-12,bx+12,by+12],fill=BADGE_BG)
        w=d.textbbox((0,0),str(i+1),font=fb)[2]
        d.text((bx-w/2,by-7),str(i+1),font=fb,fill=BADGE_TX)
        IC[i](d,x+CW//2,CY+96,t)
        ctr(d,ti,ft,x+CW//2,CY+158,TITLE); ctr(d,su,fs,x+CW//2,CY+182,SUB)
        if i<3:
            ax=x+CW+GAP//2; ay=CY+CH//2
            d.line([(ax-7,ay),(ax+5,ay)],fill=ARROW,width=2)
            d.polygon([(ax+4,ay-4),(ax+4,ay+4),(ax+10,ay)],fill=ARROW)
    return im
fr=[];du=[]
N=26
for k in range(N): fr.append(frame(k/(N-5))); du.append(90)
fr.append(frame(1.2)); du.append(1100)
pal=fr[-1].convert('P',palette=Image.ADAPTIVE,colors=64)
q=[f.quantize(palette=pal,dither=Image.NONE) for f in fr]
o='code/images/xqtl_getting_started.gif'
q[0].save(o,save_all=True,append_images=q[1:],duration=du,loop=0,optimize=True,disposal=1)
print('frames=%d %.0f KB loop=%.1fs'%(len(q),os.path.getsize(o)/1024,sum(du)/1000))
