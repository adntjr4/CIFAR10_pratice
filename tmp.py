
file_name = [   '03.09-04.57.46_PyramidalNet_a48_L110',
                '03.10-00.16.05_PyramidalNet_a48_L110_mixup',
                '04.08-23.20.36_MobileNet_13block',
                '04.09-09.15.07_MobileNet_13block_mixup',
                '04.13-23.30.51_MobileNetV2_17unit']

for i in file_name:
    before = open('./logs/print_log/%s.log'%i, 'r')
    after = open('./logs/print_log/%s_r.log'%i, 'w')

    line = before.readline()
    after.write(line.replace('\t\t\t\t\t', '\n').replace('\t\t\t', '\n'))
