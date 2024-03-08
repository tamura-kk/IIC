import subprocess

model = 'resnet50'

if __name__ == '__main__':

    subprocess.run(
        'python IIC_class.py --cfg config/{}.yaml'.format(model), shell= True
    )
    subprocess.run(
        'python IIC_class_test.py --cfg config/{}.yaml'.format(model), shell = True
    )
    subprocess.run(
        'python make_result_csv.py --cfg config/{}.yaml'.format(model), shell = True
    )
    subprocess.run(
        'python ClusterAnalysis.py --cfg config/{}.yaml'.format(model), shell = True
    )
    subprocess.run(
        'python Logistic_Recall.py --cfg config/{}.yaml'.format(model), shell = True
    )

    # for j in perplexity_list:
    #     print('make t-SNE graph with perplexity = {}'.format(j))
    #     subprocess.run(
    #         'python tsne_IICmodel.py --cfg config/{}.yaml'.format(model), shell = True
    #     )