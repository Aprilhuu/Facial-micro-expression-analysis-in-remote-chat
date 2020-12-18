# Facial-micro-expression-analysis-in-remote-chat

CSC420 balding engineers project

In this project, we proposed to build a deep learning based facial expression analysis pipeline which detects people’s emotions 
based on their facial expressions in a recorded video and generates a plot to show their trend of emotion changes.

To run our pipeline, make sure you are in the root folder of this project. Then run the command:

`python main.py`

There are some flags that can be specified along this command. Their use is specified by their names:

    --model_path, default='/content/result_resnet18/model_59.bin', type=str
    
    --model_name, default='resnet18', type=str
    
    --svm_model_path, default="/content/Facial-micro-expression-analysis-in-remote-chat/dataloader/finalized_face_detection_model.sav",
                            type=str
                            
    --device, default=0, type=int
    
    --num_classes, default=7, type=int
    
    --video_path, default='./Facial-micro-expression-analysis-in-remote-chat/data/sheldon.mp4', type=str
    
    --sample_rate, default=10, type=int
    
    --result_dir, default="./detection_results", type=str

After the pipeline is finished, generated trend plot will be saved to your directory named `detection_result.jpg`.