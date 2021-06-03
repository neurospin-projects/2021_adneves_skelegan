from PIL import Image
import glob

# Create the frames
frames = []
imgs = glob.glob("/neurospin/dico/adneves/wgan_gp/exploration2/*.png")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)

print(frames[:10])
frames=sorted(frames)
print(frames)
# Save into a GIF file that loops forever
frames[0].save('/neurospin/dico/adneves/wgan_gp/exploration2/exploration1.gif', format='GIF',
               append_images=frames,
               save_all=True,
               duration=300, loop=0)
