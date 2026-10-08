# Estimated video course / 录屏场地近似重建

这是一份依据录屏外观制作的可选视觉与几何近似，**不是同事原始场地参数**。
所有障碍尺寸、位置、颜色和相机参数均为 estimated；没有从录屏恢复真实尺度。
原 scene.xml、robot.xml、STL、机器人出生点、关节、执行器、策略、物理时间步均未修改。

内容：蓝色棋盘、青黄绿紫橙5色块、灰色台阶/平台/长坡、圆柱、碎板区、
平缓粗糙起伏区和纹路地块。碎板为35个固定薄箱体，属于形状简化；所有障碍
都有静态碰撞几何。两个起伏地块使用真正的 PNG hfield 碰撞，不是凸包网格。
高度场边界为零，粗糙区最大抬升25毫米，纹路区最大抬升35毫米。

机器人沿用原始 (0, 0, 0.8) 初始位置，周围留空。layout_estimated.json 中提供一条
驶向平缓粗糙区的几何路线，仅供人工低速操作参考，不代表控制策略已通过验证。
高台、长坡、彩块用于展示；不应把它们作为现有策略默认必须通关的考核。

本目录由 tools/generate_video_course.py 生成；该程序拒绝覆盖任何已有输出。
生成器不调用 MuJoCo、不启动仿真。场景加载与viewer验证由主任务另行完成。
相机设置意在一次显示全场及起点机器人，实际窗口比例/视距可由viewer手动调整。

复用 robot.xml 的 compiler.meshdir 时，MuJoCo 3.3 同样通过 meshdir 解析 hfield.file，
故PNG引用可能写作 ../mujoco/video_course_assets/*.png；这是相对于 ../meshes 的路径。
本地schema依据：mujoco/test/user/testdata/hfield_png.xml 与
mujoco/src/user/user_objects.cc 中 mjCHField::Compile / LoadPNG。

layout_estimated.json 记录每个碰撞体、高度场尺度、参考帧、随机种子和机器人XML散列。
