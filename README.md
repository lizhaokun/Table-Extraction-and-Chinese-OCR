# Table-Extraction-and-Chinese-OCR
Extract the outline of the table from the paper form obtained from the photo or the electronic document and recognize the text content in the outline. 从拍照得到的纸质表格或者是电子表格中检测出表格轮廓并提取出这些轮廓，对每个轮廓内的内容进行识别。

# 数据准备
创建data/img和data/test目录  
1.如果表格比较大，文字较小，需要进行以下操作：
  1）将要识别的图片数据放入data/img/目录下。
  2）根据要识别的表格内容，更改jiequ.py文件中第20行，更改需要在原图上截取的范围。
  3）运行jiequ.py
2.如果表格较小，文字较大，易识别，直接将图片放入到data/test/目录下即可

# 环境配置
