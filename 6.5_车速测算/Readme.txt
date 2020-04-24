1、下载代码。程序目录为velocity_estimate。
2、下载相关资源文件。https://pan.baidu.com/s/1HGNJFQFEF_98RE9v6yIEiQ 提取码：tcmo，“分开部署”子目录：
2.1、下载模型文件yolo.h5与mars-small128.pb，放到velocity_estimate\model_data目录下。
2.2、下载测试文件LMJK_2.mp4，放到velocity_estimate\input目录下
2.3、下载字体文件ARKai_C.ttf，放到dual_traffic_count\Font目录下。
3、在vscode环境下打开traffic_event目录，在终端下输入以下命令：
activate tensorflow-gpu
python.exe demo.py  --input input/LMJK_2.mp4 --output output/LMJK_2.mp4 --linepos 0.65 --accl 1 --stat_color 7 --label_color 3