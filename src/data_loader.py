import os
import subprocess
import zipfile

def download_and_extract_gdrive(file_id: str, dest_folder: str, filename: str = None, unzip: bool = True):
    """
    Скачивает файл с Google Drive по file_id в папку dest_folder.
    Если filename указан, сохраняет под этим именем, иначе использует имя из gdown.
    Если unzip=True и файл - zip-архив, распаковывает его в dest_folder.
    """

    import gdown  # gdown должен быть установлен в окружении

    os.makedirs(dest_folder, exist_ok=True)

    # Формируем путь для сохранения файла
    if filename is None:
        filename = file_id  # просто временное имя

    file_path = os.path.join(dest_folder, filename)

    # Если файла нет, скачиваем
    if not os.path.exists(file_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        print(f"Скачиваю с Google Drive файл {file_id} в {file_path}...")
        gdown.download(url, file_path, quiet=False)
    else:
        print(f"Файл {file_path} уже существует, пропускаем скачивание.")

    # Если нужно - распаковываем (только zip)
    if unzip and file_path.endswith('.zip'):
        print(f"Распаковываю архив {file_path} в папку {dest_folder}...")
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(dest_folder)
        print("Распаковка завершена.")