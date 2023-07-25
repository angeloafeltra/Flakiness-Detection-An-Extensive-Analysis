from pydriller import  Repository, Git
from git import Repo
import os

class Cloner:

    def __init__(self):
        self.path_folder='/Users/angeloafeltra/Documents/GitHub/Flakiness-Detection-An-Extensive-Analysis/MetricExtractor/spazioCondiviso/Repository'


    def clone_repository(self,repository,gitURL, gitSSH):

        print("STO CLONANDO LA REPOSITORY {}".format(repository))
        try:
            path_folder_repo=os.path.join(self.path_folder,'{}_{}'.format(repository,gitSSH))
            Repo.clone_from(gitURL,path_folder_repo)
            gr=Git(path_folder_repo)
            gr.checkout(gitSSH)
        except Exception as e:
            print(e)
            return False

        return True
