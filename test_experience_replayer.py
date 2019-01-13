from experience_replayer import ExperienceReplayer


print("generate a 20 samples Experience Replayer with priority and load 3 tuple samples and a 2 string sample")
exr = ExperienceReplayer(20, 0.001, 0.2)

exr.store("rare string")
exr.store((1, 2), 1)
exr.store((3, 4, 5, 6), 2)
exr.store((7, 8, 9), 3)
exr.store("common string", 10)

print("draw 10 samples. I expect to never see the same sample in a list and to see the cmmon string often and the rare string rarely")
for i in range(10):
    print(exr.draw(3))

out = exr.draw(2)

print("draw a number of samples exceeding saved samples. I expect return None, thus to print nothing")
print(out)

