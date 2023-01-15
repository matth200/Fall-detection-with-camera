<h1>Fall-detection-with-camera</h1>

<h3>Presentation</h3>

With this program, you can detect people who are falling and start an action when it happens<br/>
You have detection in live on people<br/>

![My Images](documentation/7.png)

To better detection I had a timer when it exceeds a certain time, it triggers an action ( by default it just rings an alarm ) <br/>
we can see the duration of the fall on the screen's top-left corner<br/>

![My Images](documentation/9.png)

You can even detect multiple person on the same image and even in live with streaming camera

![My Images](documentation/4.png)

![My Images](documentation/11.png)

![My Images](documentation/14.png)

<h3>What do you need</h3>

You will need tensorflow, numpy, opencv <br/>
and also python3, python3-pip<br/>

<code> ~$ sudo apt-get install python3 python3-pip</code><br/>
<code> ~$ sudo python3 -m pip install numpy tensorflow opencv-python</code><br/>


And you also need to launch the installation script <br/>
<code> ~$ chmod +x install.sh</code><br/>
<code> ~$ ./install.sh </code><br/>

<h3>How to use it</h3>

You have multiple file, you can launch<br/>
but mainly you can use detection.py and detection_timer.py<br/>
The difference between the two is just the number of people you can detect with the one detection.py you can only have one person on the detection<br/>
but you have less needs in computation and the one detection_timer.py you can detect multiple with no problem buuttt you need a little bit more of computation<br/>


So go to the directory where are all the files<br/>

<code> ~$ chmod +x detection.py detection_timer.py</code><br/>

Then you launch whatever the file you want<br/>
<code> ~$ ./detection.py</code><br/>
ou<br/>
<code> ~$ ./detection_timer.py</code><br/>

<h3>How it works</h3>

It uses the algorithm Yolo to detect shape of people and once it detects all the shapes, the shapes of people are passing through an other algorithm<br/>
which I have made and can tell you if those shapes are falling or not<br/>

Algorithm:<br/>

![My Images](documentation/15.png)

<h3>Why</h3>

I made it for my TIPE which are an exam from Classe Préparatoire to pass some exams<br/>
You can have the presentation in the documentation folder


