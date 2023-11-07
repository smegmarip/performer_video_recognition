"""
The ProgressBar class provides a convenient way to create and update a progress bar in Python using
the tqdm library.
"""
import tqdm


class ProgressBar:
    """
    Progress Bar
    """

    def __init__(self) -> None:
        """
        Constructor
        """
        self.progress_bar = None

    def init_progressbar(self, total_frames):
        """
        The function initializes a progress bar with a specified total number of frames.

        :param total_frames: The total number of frames that the progress bar will track
        """
        self.progress_bar = tqdm(total=total_frames)

    def update_progressbar(self):
        """
        The function updates a progress bar by incrementing its value by 1.
        """
        self.progress_bar.update(1)

    def close_progressbar(self):
        """
        The function `close_progressbar` closes a progress bar.
        """
        self.progress_bar.close()
