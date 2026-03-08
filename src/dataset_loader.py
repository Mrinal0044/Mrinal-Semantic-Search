import os

def load_dataset(data_path):

    documents = []
    labels = []

    for category in os.listdir(data_path):

        category_path = os.path.join(data_path, category)

        # skip non-folder files
        if not os.path.isdir(category_path):
            continue

        for filename in os.listdir(category_path):

            file_path = os.path.join(category_path, filename)

            try:
                with open(file_path, "r", encoding="latin-1") as f:
                    text = f.read()

                if text.strip():
                    documents.append(text)
                    labels.append(category)

            except Exception:
                continue

    return documents, labels