import matplotlib.pyplot as plt
import numpy as np

label_to_number = {6: 20, 3: 10, 4: 0, 5: -10, 2: -20, 0: -30, 1: -40}
label_to_text = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}


def plot_trend(number_of_frames, emotion_labels, input_video_name="Sheldon.mp4", save_path="./"):
    """
    Displaying a trend plot to show detection result

    :param input_video_name: Name of the input video, used to set plot title
        :type input_video_name: str
    :param save_path: Path to save trend plot
        :type save_path: str
    :param number_of_frames: An integer indicating number of frames detected
        :type number_of_frames: int
    :param emotion_labels: A list of integer emotion labels corresponding to each frame
        :type emotion_labels: list
    :return:
    """
    assert number_of_frames == len(emotion_labels), "Number of frames should equal to number of emotion labels!"
    frame_numbers = np.arange(number_of_frames)
    x_labels = ["frame " + str(ind) for ind in frame_numbers]
    y_labels = [label_to_number[i] for i in emotion_labels]
    plt.yticks([])
    plt.ylim(-50, 30)
    plt.axhline(y=0, color='k')
    plt.bar(x_labels, y_labels)

    axes2 = plt.twinx()
    axes2.plot(x_labels, y_labels, color='red', lw=4)
    axes2.set_ylim(-50, 30)
    axes2.set_yticks([])

    for index, value in enumerate(emotion_labels):
        if y_labels[index] > 0:
            position = "bottom"
            offset = 2.5
        else:
            position = "top"
            offset = -2.5

        plt.text(index, y_labels[index] + offset, label_to_text[value],
                 ha='center', va=position, fontsize='large', weight='bold')

    plt.annotate("negative emotion", xy=(0, 0.2), xytext=(-10, 0), rotation=90,
                 xycoords="axes fraction", textcoords="offset points", ha="center")
    plt.annotate("positive emotion", xy=(0, 0.68), xytext=(-10, 0), rotation=90,
                 xycoords="axes fraction", textcoords="offset points", ha="center")

    if input_video_name:
        plt.title("Trend in emotion change detected for " + input_video_name)
    plt.savefig(save_path + "detection_result.jpg", format='jpg', dpi=400)
    plt.show()
    plt.close()


if __name__ == "__main__":
    plot_trend(7, [6, 3, 5, 5, 0, 0, 0])
