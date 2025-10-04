import gradio as gr
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model('model.h5')

def recognize_digit(image):
    if image is not None:

        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255
        
        prediction = model.predict(image)
        
        # Return dictionary of probabilities
        return {str(i): float(prediction[0][i]) for i in range(10)}
    else:
        return ''
    
    
    # Gradio interface
iface = gr.Interface(
    fn=recognize_digit,
    # inputs=gr.Image(type="numpy", image_mode='L', sources='clipboard'),
    # inputs=gr.ImageEditor(shape=(28, 28), image_mode='L'),
    # inputs=gr.Image(shape=(28, 28), image_mode='L', invert_colors=True, sources='canvas')
    outputs=gr.Label(num_top_classes=3),
    live=True
)
iface.launch()





