import os
import labelimg
import inspect

# 1. Get the actual folder where labelimg is installed
labelimg_path = os.path.dirname(inspect.getfile(labelimg))
print(f"LabelImg is located at: {labelimg_path}")

# 2. Look for canvas.py in the 'libs' subfolder
canvas_path = os.path.join(labelimg_path, 'libs', 'canvas.py')

if os.path.exists(canvas_path):
    print(f"SUCCESS! Found canvas.py at: {canvas_path}")

    with open(canvas_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Apply the patches to fix the 'float to int' crash
    # Patch 1: drawRect
    content = content.replace('p.drawRect(left_top.x(), left_top.y(), rect_width, rect_height)',
                              'p.drawRect(int(left_top.x()), int(left_top.y()), int(rect_width), int(rect_height))')

    # Patch 2: drawLine (Vertical)
    content = content.replace('p.drawLine(self.prev_point.x(), 0, self.prev_point.x(), self.pixmap.height())',
                              'p.drawLine(int(self.prev_point.x()), 0, int(self.prev_point.x()), int(self.pixmap.height()))')

    # Patch 3: drawLine (Horizontal)
    content = content.replace('p.drawLine(0, self.prev_point.y(), self.pixmap.width(), self.prev_point.y())',
                              'p.drawLine(0, int(self.prev_point.y()), int(self.pixmap.width()), int(self.prev_point.y()))')

    with open(canvas_path, 'w', encoding='utf-8') as f:
        f.write(content)
    print("Patch applied successfully. LabelImg should no longer crash on 'w'!")
else:
    print("Still can't find canvas.py. Let's try the manual Windows search.")