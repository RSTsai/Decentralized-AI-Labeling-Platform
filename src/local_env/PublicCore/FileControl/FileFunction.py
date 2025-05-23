from os import makedirs, path, pardir, listdir



def GetFileNameWithoutExtension(filename):
    file_name = path.basename(filename)
    file_name, file_extension = path.splitext(file_name)
    return file_name


def CheckDirPath(filePath):
    if path.isfile(filePath):
        dirPath= path.dirname(filePath)
    else:
        dirPath = filePath   
    CheckDir(dirPath)
    return dirPath


def CheckDir(dirPath):
    try:      
        # Check if the directory exists
        if not path.exists(dirPath):
            # If not, create the directory
            makedirs(dirPath)
            print(f"Directory '{dirPath}' created successfully.")
            success = True
        else:
            # If the directory already exists, print a message
            #print(f"Directory '{dirPath}' already exists.")
            success = True

    except PermissionError:
        # If a PermissionError occurs, print a message
        print(f"Permission error: Unable to create directory '{dirPath}'.")
        success = False
    except Exception as e:
        # If there was a different error, print the error message
        print(f"An error occurred: {e}")
        success = False
    finally:
        # This block will run regardless of whether an exception was raised or not
        return success


def CheckFile(filePath):
    try:
        if path.exists(filePath):
            return True
        else:
            print(f"path not exists:\n{filePath}")
            
            parentDirPath = path.abspath(path.join(filePath, pardir))
            filesInParentDir = [f for f in listdir(parentDirPath) if path.isfile(path.join(parentDirPath, f))]
            
            print(f"\nDirFile:\n{parentDirPath}")
            [print(_filename) for _filename in filesInParentDir]
            return False
        
    except FileExistsError:
        print("FileExistsError")
        return False
