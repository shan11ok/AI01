1、下载代码。程序目录为traffic_event。
2、下载相关资源文件。https://pan.baidu.com/s/1Hx6tMUpykv48tf7qQL99gQ 提取码：u3gn，“分开部署”子目录：
2.1、下载模型文件yolo.h5与mars-small128.pb，放到traffic_event\model_data目录下。
2.2、下载测试文件p1.mp4,s1.mp4，放到traffic_event\input目录下。
2.3、下载字体文件ARKai_C.ttf，放到traffic_event\Font目录下。
3、在vscode环境下打开traffic_event目录，在终端下输入以下命令：
activate tensorflow-gpu
python.exe demo.py  --input input/s1.mp4 --output output/s1.mp4 --direction 1 --stat_color 7 --label_color 3
python.exe demo.py  --input input/p1.mp4 --output output/p1.mp4 --direction 1 --stat_color 7 --label_color 3

