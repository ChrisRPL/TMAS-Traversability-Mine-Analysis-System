#!/bin/bash
# Setup Blender for synthetic data generation
# Installs Blender 4.0+ with Python scripting support

set -e

BLENDER_VERSION="4.0.2"
BLENDER_URL="https://download.blender.org/release/Blender4.0/blender-${BLENDER_VERSION}-linux-x64.tar.xz"
INSTALL_DIR="${HOME}/.local/blender"

echo "========================================="
echo "Blender Setup for TMAS Synthetic Data"
echo "========================================="
echo "Version: ${BLENDER_VERSION}"
echo "Install directory: ${INSTALL_DIR}"
echo ""

# Create installation directory
mkdir -p "${INSTALL_DIR}"

# Download Blender
echo "Downloading Blender ${BLENDER_VERSION}..."
cd /tmp
if [ ! -f "blender-${BLENDER_VERSION}-linux-x64.tar.xz" ]; then
    wget "${BLENDER_URL}" -O "blender-${BLENDER_VERSION}-linux-x64.tar.xz"
else
    echo "Blender archive already downloaded"
fi

# Extract Blender
echo "Extracting Blender..."
tar -xf "blender-${BLENDER_VERSION}-linux-x64.tar.xz"

# Move to install directory
echo "Installing to ${INSTALL_DIR}..."
rm -rf "${INSTALL_DIR}/blender-${BLENDER_VERSION}-linux-x64"
mv "blender-${BLENDER_VERSION}-linux-x64" "${INSTALL_DIR}/"

# Create symlink
echo "Creating symlink..."
ln -sf "${INSTALL_DIR}/blender-${BLENDER_VERSION}-linux-x64/blender" "${INSTALL_DIR}/blender"

# Add to PATH
BLENDER_BIN="${INSTALL_DIR}"
echo ""
echo "========================================="
echo "Add Blender to PATH by adding this line to ~/.bashrc or ~/.zshrc:"
echo "export PATH=\"${BLENDER_BIN}:\$PATH\""
echo "========================================="
echo ""

# Test Blender installation
echo "Testing Blender installation..."
"${INSTALL_DIR}/blender" --version

# Get Blender Python path
BLENDER_PYTHON=$("${INSTALL_DIR}/blender" --background --python-expr "import sys; print(sys.executable)" 2>/dev/null | tail -1)
echo "Blender Python: ${BLENDER_PYTHON}"

# Install Python dependencies in Blender's Python
echo ""
echo "Installing Python dependencies in Blender..."
"${BLENDER_PYTHON}" -m ensurepip
"${BLENDER_PYTHON}" -m pip install --upgrade pip
"${BLENDER_PYTHON}" -m pip install numpy opencv-python

# Test headless rendering
echo ""
echo "Testing headless rendering..."
cat > /tmp/test_render.py << 'EOF'
import bpy

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Create simple test scene
bpy.ops.mesh.primitive_cube_add(location=(0, 0, 0))
bpy.ops.object.light_add(type='SUN', location=(5, 5, 10))

# Setup camera
bpy.ops.object.camera_add(location=(7, -7, 5))
camera = bpy.context.object
camera.rotation_euler = (1.1, 0, 0.785)
bpy.context.scene.camera = camera

# Setup render settings
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.render.resolution_x = 256
bpy.context.scene.render.resolution_y = 256
bpy.context.scene.render.filepath = '/tmp/test_render.png'

# Render
bpy.ops.render.render(write_still=True)
print("Test render complete: /tmp/test_render.png")
EOF

"${INSTALL_DIR}/blender" --background --python /tmp/test_render.py

if [ -f "/tmp/test_render.png" ]; then
    echo ""
    echo "========================================="
    echo "SUCCESS: Blender setup complete!"
    echo "Test render created: /tmp/test_render.png"
    echo ""
    echo "Blender executable: ${INSTALL_DIR}/blender"
    echo "Blender Python: ${BLENDER_PYTHON}"
    echo "========================================="
else
    echo ""
    echo "ERROR: Test render failed"
    exit 1
fi
