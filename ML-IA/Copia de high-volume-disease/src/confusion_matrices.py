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
    
    #fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 6))
    fig, axes = plt.subplots(2, 2, figsize=(3 * len(models), 8))

    axes = axes.ravel()
    axes = axes.flatten()  # ✅ Aplanar la matriz de subplots

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
    plt.savefig(output_path, dpi=300)
    plt.show()
    print(f'✅ Imagen combinada guardada en: {output_path}')