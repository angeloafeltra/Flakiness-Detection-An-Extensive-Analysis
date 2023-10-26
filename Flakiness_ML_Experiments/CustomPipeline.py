from imblearn.pipeline import Pipeline

class CustomPipeline:

    nome=None
    preProcessing=None
    clf=None
    pipeline=None

    def __init__(self,nome,preProcessing,clf):
        self.nome=nome
        self.preProcessing=preProcessing
        self.clf=clf
        steps=[]
        for item in self.preProcessing:
            steps.append(item)
        steps.append(('clf',clf))
        self.pipeline=Pipeline(steps = steps)

    def getNome(self): return self.nome
    def getPreProcessing(self): return self.preProcessing
    def getClassificatore(self): return self.clf
    def getPipeline(self): return self.pipeline

    def toString(self):
        str='PipelineName: {}\nPipeline: '.format(self.nome)
        for step in self.pipeline.steps:
            if step[0]!='clf':
                str=str+step[0]+"_"
            else:
                str=str+step[1].__class__.__name__

        return str