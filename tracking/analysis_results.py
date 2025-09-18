import _init_paths
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []

'''
cd /home/dell/WorkSpace/Tracking_by_NL/All-in-One/tracking/
conda activate python38   
python analysis_results.py
'''

# dataset_name = 'lasot'
# dataset_name = 'lasotext'
# dataset_name = 'tnl2k'
dataset_name = 'otb99lang'
# dataset_name = 'webuav3m'


"""stark"""
# trackers.extend(trackerlist(name='stark_s', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-S50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST50'))
# trackers.extend(trackerlist(name='stark_st', parameter_name='baseline_R101', dataset_name=dataset_name,
#                             run_ids=None, display_name='STARK-ST101'))
"""TransT"""
# trackers.extend(trackerlist(name='TransT_N2', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N2', result_only=True))
# trackers.extend(trackerlist(name='TransT_N4', parameter_name=None, dataset_name=None,
#                             run_ids=None, display_name='TransT_N4', result_only=True))
"""pytracking"""
# trackers.extend(trackerlist('atom', 'default', None, range(0,5), 'ATOM'))
# trackers.extend(trackerlist('dimp', 'dimp18', None, range(0,5), 'DiMP18'))
# trackers.extend(trackerlist('dimp', 'dimp50', None, range(0,5), 'DiMP50'))
# trackers.extend(trackerlist('dimp', 'prdimp18', None, range(0,5), 'PrDiMP18'))
# trackers.extend(trackerlist('dimp', 'prdimp50', None, range(0,5), 'PrDiMP50'))

"""ostrack"""
trackers.extend(trackerlist(name='ostrack', parameter_name='vitb_256_mae_ce_32x4_ep300', dataset_name=dataset_name,
                            run_ids=None, display_name='OSTrack256'))

dataset = get_dataset(dataset_name)
# dataset = get_dataset('otb', 'nfs', 'uav', 'tc128ce')
# plot_results(trackers, dataset, 'OTB2015', merge_results=True, plot_types=('success', 'norm_prec'),
#              skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'norm_prec', 'prec'))
# print_results(trackers, dataset, 'UNO', merge_results=True, plot_types=('success', 'prec'))


'''
seamtrack_track
# 30
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.04      | 81.43      | 69.79      | 76.55        | 79.49             |

50 
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.43      | 81.93      | 70.43      | 77.38        | 80.14             

tihuanjiyimoban
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 69.22      | 81.01      | 67.20      | 74.44        | 78.80             |

==================  以上没有训练语言编码部分 ===========================
==================  训练语言编码部分 ===========================
sematrack 训练语言编码模块 20轮
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.21      | 82.27      | 69.59      | 77.45        | 80.32             |

50轮
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.72      | 82.34      | 70.64      | 77.83        | 80.53             |

替换记忆模板
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 69.36      | 81.22      | 67.74      | 75.13        | 79.25             |

记忆模板(视频级替换)
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.72      | 82.21      | 70.78      | 77.89        | 80.61             |

记忆模板(视频级替换)_语言更新
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.36      | 81.77      | 70.53      | 77.25        | 80.09             |

==================  训练语言编码部分 + 对齐  微调30轮  ===========================
加 语言，记忆模板
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 69.85      | 81.34      | 69.35      | 76.67        | 79.68             |

记忆模板(视频级替换)
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 69.85      | 81.31      | 69.34      | 76.64        | 79.57             |





test_无对齐_记忆模板置信度7_轮次50
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.30      | 81.72      | 70.30      | 77.20        | 79.99             |
tnl2k           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 65.40      | 76.54      | 62.45      | 70.11        | 83.18             |
==================  训练语言编码部分 + 对齐  对齐改进+权重+增加相似矩阵差距  ==========================
test_wuduiqi_记忆模板置信度7_轮次50
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.30      | 81.72      | 70.30      | 77.20        | 79.99             |
test_对齐v2_记忆模板置信度7_轮次1
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.57      | 83.38      | 71.19      | 78.96        | 81.72             |
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.57      | 90.69      | 55.74      | 94.11        | 87.39             |
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
test_对齐v2_记忆模板置信度7_轮次1_ce_0.7
tnl2k           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 64.90(低)      | 75.92      | 61.90      | 69.62        | 82.68             |
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.57（高）      | 83.38      | 71.19      | 78.96        | 81.72             |
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.57(低)      | 90.69      | 55.74      | 94.11        | 87.39             |
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
test_对齐v2_记忆模板置信度7_轮次1_ce_08
tnl2k           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 65.56(低)      | 76.68      | 62.85      | 70.53        | 83.37             |

otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.33(低)      | 90.07      | 55.71      | 93.58        | 87.37             |
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
test_对齐v2_记忆模板置信度7_轮次1_ce_09
tnl2k           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 65.80（最高）（比SemaTrack高）      | 76.89      | 63.26      | 70.62        | 83.57             |
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.03(低)（比SemaTrack低，第二）    | 82.70      | 70.88      | 78.21        | 81.04             |
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.40(低)（比SemaTrack低，第三，第二可去）       | 90.34      | 56.14      | 93.67        | 87.27             |
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
test_对齐v2_记忆模板置信度7_轮次1_ce_1（保留率为1,即不删除）
tnl2k           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 65.63      | 76.66      | 63.18      | 70.42        | 83.33             |
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.09      | 82.87      | 71.01      | 78.34        | 81.12             |
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.63      | 90.52      | 56.21      | 94.29        | 87.87             |
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
987 
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.10      | 90.07      | 55.33      | 93.49        | 87.17             |
789
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.66      | 90.65      | 56.15      | 94.16        | 87.78             |
tnl2k           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 65.32      | 76.44      | 62.52      | 70.20        | 83.17             |
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


test_对齐v2_记忆模板置信度6_轮次1
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.23      | 83.09      | 70.79      | 78.71        | 81.36             |
test_对齐v2_记忆模板置信度8_轮次1
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.76      | 82.47      | 70.46      | 77.97        | 80.81             |
test_对齐v2_记忆模板置信度7_轮次10
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 69.98      | 81.53      | 69.70      | 77.06        | 79.91             |
test_对齐v2_记忆模板置信度7_轮次2
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.54      | 82.23      | 70.29      | 77.84        | 80.63             |

test_对齐v2_记忆模板置信度7_语言更新_CSDGModule_轮次1
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.17      | 82.95      | 70.82      | 78.54        | 81.23             |

test_对齐v2_记忆模板置信度7_CSDG_update_language_有提示_轮次1
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.16      | 82.98      | 70.79      | 78.46        | 81.18             |

test_对齐v2_记忆模板置信度7_拼接_update_language_轮次1
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.87      | 82.58      | 70.46      | 78.20        | 80.85             |

test_对齐v2_记忆模板置信度7_拼接_update_language_模板加当前帧_轮次1_语言提示 带初始模板-初始语言提示
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.35      | 83.12      | 71.15      | 78.82        | 81.42             |
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.36      | 90.56      | 55.78      | 93.85        | 87.19             |
test_对齐v2_记忆模板置信度7_拼接_update_language_模板加当前帧_轮次1_提示不带语言
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.36      | 90.56      | 55.78      | 93.85        | 87.19             |
test_对齐v2_记忆模板置信度7_拼接_update_language_当前帧_轮次1
otb99lang       | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.36      | 90.56      | 55.78      | 93.85        | 87.19             |

test_对齐v2_记忆模板置信度7_拼接_update_language_模板加当前帧_轮次1_语言提示  不带初始模板-更新后的语言提示(f'Question: The previous description of the {object_class} was: {language} 'f'Given the template images showing the {object_class}\'s stable appearance and the current frame showing the updated scene, 'f'Based on the current image, generate an updated and consistent description of the {object_class}. Answer:'
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.03      | 82.79      | 70.69      | 78.33        | 81.14             |
test_对齐v2_记忆模板置信度7_拼接_update_language_模板del0加当前帧_轮次1
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.82      | 82.46      | 70.51      | 78.12        | 80.72             |

test_对齐v2_记忆模板置信度7_拼接_update_language_模板加当前帧_轮次1_类提示
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 69.92      | 81.41      | 69.41      | 76.98        | 79.66             |
test_对齐v2_记忆模板置信度7_拼接_update_language_当前帧每帧更新_轮次1
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.28      | 83.11      | 70.86      | 78.59        | 81.31             |


test_对齐v2_记忆模板置信度7_拼接_update_language_模板加当前帧_i2d_轮次1_类提示
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 71.20      | 82.97      | 70.76      | 78.61        | 81.28             |

test_对齐v2_记忆模板置信度7_拼接_update_language_上帧结果加当前帧_i2d_轮次1_类提示
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.84      | 82.61      | 70.43      | 78.12        | 80.84             |

test_对齐v2_记忆模板置信度7_拼接_update_language_模板_i2d_轮次1_类提示
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.69      | 82.45      | 70.22      | 78.02        | 80.74             |

test_对齐v2_记忆模板置信度7_拼接_update_language_当前帧每帧更新_i2d_轮次1_类提示
lasot           | AUC        | OP50       | OP75       | Precision    | Norm Precision    |
OSTrack256      | 70.94      | 82.73      | 70.46      | 78.34        | 80.93             |


'''




