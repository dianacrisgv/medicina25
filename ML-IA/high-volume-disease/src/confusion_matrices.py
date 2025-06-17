import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def show_and_save_confusion_matrices(output_path='./models/matrices_combinadas.png'):
    models = {
        'Regresión Logística': './models/conf_matrix_logistic.png',
        'Árbol de Decisión': './models/conf_matrix_tree.png',
        'Máquina Soporte Vectorial': './models/conf_matrix_svm.png',
        'Random Forest': './models/conf_matrix_rf.png' 
        
    }
    # 'XGBoost': './models/conf_matrix_xgboost.png'

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))

    # Asegura que axes sea una lista aunque haya solo una imagen
    if len(models) == 1:
        axes = [axes]

    for ax, (name, path) in zip(axes, models.items()):
        if os.path.exists(path):
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(name, fontsize=14)
        else:
            ax.set_visible(False)
            print(f'❌ Imagen no encontrada: {path}')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f'✅ Imagen combinada guardada en: {output_path}')