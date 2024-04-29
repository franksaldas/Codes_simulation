import ffmpeg
(
    ffmpeg
    .input('*.png', pattern_type='glob', framerate=10)
    .filter('scale', size='4k3840', force_original_aspect_ratio='increase')
    .output('Halo0_proj01.mp4')
    .run()
)
