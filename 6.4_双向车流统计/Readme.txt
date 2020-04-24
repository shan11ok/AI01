1、下载代码。程序目录为dual_traffic_count。
2、下载相关资源文件。https://pan.baidu.com/s/1WAwX__DExrrhoqu93Vo4Vg 提取码：ds4e，“分开部署”子目录：
2.1、下载模型文件yolo.h5与mars-small128.pb，放到dual_traffic_count\model_data目录下。
2.2、下载测试文件SD_1.mp4,SD_1.mp4，放到dual_traffic_count\input目录下
2.3、下载字体文件ARKai_C.ttf，放到dual_traffic_count\Font目录下。
3、在vscode环境下打开traffic_event目录，在终端下输入以下命令：
activate tensorflow-gpu
python.exe demo.py  --input input/SD_1.mp4 --output output/SD_1.mp4 --linepos 0.65 --direction 2 --accl 1 --stat_color 7 --label_color 3
python.exe demo.py  --input input/SD_3.mp4 --output output/SD_3.mp4 --linepos 0.5 --direction 1 --accl 1 --stat_color 7 --label_color 3