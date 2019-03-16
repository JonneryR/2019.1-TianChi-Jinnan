# 2019.1-TianChi-Jinnan
天池津南，solo复赛6/2682，总排名11/2682。  
运行model_FuSai.py，得到的Basyen_stacking.csv是结果；  
然后和鱼的开源融合，开源占0.2-0.4都可以。   
感谢鱼的开源    
整体思路:  
1.数据预处理，主要借鉴鱼的思路    
2.cross特征    
3.B14的分箱特征(具体作用需要尝试)    
4.add_div特征，对数据进行加法和除法的操作  
5.count特征    
6.lgb，xgb最后bayes融合。        
