function model=toydata
    num=5;
    noise_level=0.01;
    x1=rand(num,1)*10;
    x2=rand(num,1)*10;
    x3=rand(num,1)*10;
    y1=3*x1'+10+rand(1,num)*noise_level;
    y2=-3*x2'-5+rand(1,num)*noise_level;
    y3=1+rand(1,num)*noise_level;
    data=cell(1,3);
    data{1}=x1;
    data{2}=x2;
    data{3}=x3;
    label=cell(1,3);
    label{1}=y1;
    label{2}=y2;
    label{3}=y3;
    model=MTRL(data,label,'linear',0,0.1,0.1);