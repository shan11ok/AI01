1�����ش��롣����Ŀ¼Ϊtraffic_event��
2�����������Դ�ļ���https://pan.baidu.com/s/1Hx6tMUpykv48tf7qQL99gQ ��ȡ�룺u3gn�����ֿ�������Ŀ¼��
2.1������ģ���ļ�yolo.h5��mars-small128.pb���ŵ�traffic_event\model_dataĿ¼�¡�
2.2�����ز����ļ�p1.mp4,s1.mp4���ŵ�traffic_event\inputĿ¼�¡�
2.3�����������ļ�ARKai_C.ttf���ŵ�traffic_event\FontĿ¼�¡�
3����vscode�����´�traffic_eventĿ¼�����ն��������������
activate tensorflow-gpu
python.exe demo.py  --input input/s1.mp4 --output output/s1.mp4 --direction 1 --stat_color 7 --label_color 3
python.exe demo.py  --input input/p1.mp4 --output output/p1.mp4 --direction 1 --stat_color 7 --label_color 3

