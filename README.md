# yuan
opencv3自带8种目标跟踪算法，CRST效果最好，速度稍慢。
还有一种基于深度学习的tracker——GOTURN Tracker，需额外模型，待研究。
 opencv8种目标跟踪方法
1、MedianFlow——无人机拍摄视频
      报错方面表现好，对快速运动或快速移动模型会失效。
2、boosting tracker和Haar cascades:机器学习算法相同（最早的tracker）
        缺点：速度慢、表现不好
3、MIL tracker：精确一点，失败率高
4、KCF tracker：快，有遮挡不佳【速度】
5、CSRT tracker：比KCF精确，比KCF慢。【精确度】
6、TLD：报错很多
7、MOSSE Tracker：速度快，准确率比CSRT和KCF低
8、GOTURN Tracker：深度学习基础的目标检测器，需额外模型。

GOTURN Tracker参考资料：
https://github.com/spmallick/learnopencv/tree/master/GOTURN

https://blog.csdn.net/zmdsjtu/article/details/81630366
