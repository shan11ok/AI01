1�����ش��롣����Ŀ¼Ϊvelocity_estimate��
2�����������Դ�ļ���https://pan.baidu.com/s/1HGNJFQFEF_98RE9v6yIEiQ ��ȡ�룺tcmo�����ֿ�������Ŀ¼��
2.1������ģ���ļ�yolo.h5��mars-small128.pb���ŵ�velocity_estimate\model_dataĿ¼�¡�
2.2�����ز����ļ�LMJK_2.mp4���ŵ�velocity_estimate\inputĿ¼��
2.3�����������ļ�ARKai_C.ttf���ŵ�dual_traffic_count\FontĿ¼�¡�
3����vscode�����´�traffic_eventĿ¼�����ն��������������
activate tensorflow-gpu
python.exe demo.py  --input input/LMJK_2.mp4 --output output/LMJK_2.mp4 --linepos 0.65 --accl 1 --stat_color 7 --label_color 3