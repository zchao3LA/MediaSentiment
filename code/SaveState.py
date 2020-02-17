import datetime
import os
import shutil

#Path to data
repo_root = '/Users/denalimolitor/git_repos/Sentiment'
"""
Saves a copy of the current code so that images can be easily recreated and
so that the parameters used to generate images are obvious
"""

def get_output_dir(basename="", dest_folder = "results"):
    """
    Create a directory
        DeannaResearch > 'dest_folder' > DATE_TIME
    duplicate current Code into
        DeannaResearch > 'dest_folder' > DATE_TIME > code_stash
    return the path to the DATE_TIME directory
    """

    suffix = datetime.datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    if basename:
        dirname = "_".join([basename, suffix])
    else:
        dirname = suffix

    dirpath = os.path.join(repo_root, dest_folder, dirname)
    os.makedirs(dirpath)

    shutil.copytree(".", os.path.join(dirpath, "code_stash"))
    return dirpath
