# 🚀 GUÍA COMPLETA GIT & GITHUB
## 📌 Creación y Organización de Repositorios

---

## 🎯 **1. CREAR REPOSITORIO EN GITHUB**
1. Entrar a [GitHub.com](https://github.com)
2. Click en "+" → "New repository"
3. **Nombre:** `Programming` (o el que prefieras)
4. **Importante:** NO marcar:
   - [ ] Add README
   - [ ] Add .gitignore  
   - [ ] Add license
5. Click "Create repository"

---

## 💻 **2. CONFIGURACIÓN INICIAL LOCAL**
```bash
# Navegar a directorio de trabajo
cd Documents/Programming

# Inicializar Git
git init

# Configurar usuario
git config user.name "TuNombre"
git config user.email "tu@email.com"

# Conectar con repositorio remoto
git remote add origin https://github.com/tuusuario/Programming.git

##  **3. CREAR CARPETAS**
mkdir -p NOMBRE DE LA CARPETA

##  **4. ARCHIVOS ESCENCIALES**
# Crear README y .gitignore
touch README.md
touch .gitignore

# Contenido de .gitignore:
echo ".DS_Store" >> .gitignore
echo "**/.DS_Store" >> .gitignore  
echo "*.log" >> .gitignore
echo "node_modules/" >> .gitignore
echo "__pycache__/" >> .gitignore

##. **5. PRIMER COMMIT Y PUSH**
git add .
git status  # Verificar
git commit -m "Commit inicial: estructura organizada"
git push -u origin main

##. **6. AÑADIR PROGRAMAS EXISTENTES**
mv ~/ruta/al/proyecto/ Python/Proyectos/
git add .
git commit -m "Agregar proyecto X"
git push

##  **7. COMANDOS BÁSICOS DIARIOS**
git status          # Ver estado
git add .           # Agregar todo
git commit -m "msg" # Hacer commit  
git push            # Subir cambios
git pull            # Bajar cambios

##  **8. SOLUCIÓN DE PROBLEMAS COMUNES
#Error: "fatal: refusing to merge unrelated histories"
git pull origin main --allow-unrelated-histories

#Error: "Updates were rejected"
git pull origin main
git push

#Eliminar archivos no deseados
git rm --cached archivo.txt
git rm -r --cached carpeta/

##. **9. COMANDOS DE AYUDA**
git help           # Ayuda general
git config --list  # Ver configuración
git log --oneline  # Historial de commits
git diff           # Ver cambios específicos