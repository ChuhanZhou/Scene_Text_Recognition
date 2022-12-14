
var file = document.getElementById('file');
var image = document.querySelector("#img");
var text = document.querySelector('#text')
var button = document.getElementById("btn");
var bigImage = document.querySelector('#bigimg');
var fileD;

file.onchange = function () {
        var fileData = this.files[0];//get the first file of the FileList
        var pettern = /^image/;

        //console.info(fileData.type)

        if (!pettern.test(fileData.type)) {
            alert("the format of picture is not correct");
            return;
        }
        var reader = new FileReader();
        reader.readAsDataURL(fileData);//read file content

        reader.onload = function (e) {
            //console.log(e);
            console.log(this.result);
            image.setAttribute("src", this.result);
            image.style.display = "block";
            fileD = this.result;
        }
    }

$(document).ready(function() {
    $('#btn').click(function () {
        console.log('click button >>>>')
            if (fileD!=="" && fileD!=null)
            {
                btn.disabled = true;
                downloadBtn.disabled = true;
                copyBtn.disabled = true;
                btn.style.pointerEvents = "none";
                btn.style.display = "none";
                load.style.display = "block";
                file.style.display = "none";
                var formFile = new FormData($('#fileForm')[0])
                console.log("formFile is "+formFile)
                setTimeout(alertFunc, 1000*60*3);      　//三秒之后调用alertFunc函数
                function alertFunc() {
                    alert("Please wait for the result");
                }
                $.ajax({
                    url: '/upload',
                    type: 'POST',
                    data: formFile,
                    processData: false,
                    contentType: false,
                    success: function (data) {
                        console.log("post successfully");
                        text.value = data.text;
                        if (data.outPath!=="" && data.outPath !=null )
                        {
                            //var strBase64 = Base64.decode(data.outPath);
                            url = "data:image/png;base64,";
                            image.setAttribute("src", url+data.outPath);
                        }

                        image.style.display = "block";
                        btn.disabled = false;
                        btn.style.pointerEvents = "auto";
                        downloadBtn.disabled = false;
                        copyBtn.disabled = false;
                        btn.style.display = "block";
                        load.style.display = "none";
                        file.style.display = "block";
                        clearTimeout();

                },
                error: function ()
                {
                    console.log("post fail");
                }
            })
            }
            else {
                console.log("NOT INSERT PICTURE");
                alert("PLEASE UPLOAD IMAGE");
            }


    })


    $('#copyBtn').click(function () {
        if (text.value!==""&&text.value!=null)
        {
            text.select();
            document.execCommand("Copy");
            alert("copy successfully！");
        }
        else {
            console.log("no text now");
        }
    })

    $('#downloadBtn').click(function () {
        if (text.value!==""&&text.value!=null)
        {
            var blob = new Blob([text.value], { type: "text/plain"});
            var anchor = document.createElement("a");
            anchor.download = "text.txt";
            anchor.href = window.URL.createObjectURL(blob);
            anchor.target ="_blank";
            anchor.style.display = "none"; // just to be safe!
            document.body.appendChild(anchor);
            anchor.click();
            document.body.removeChild(anchor);
        }
        else {
            console.log("no text now");
        }

    })

    $('#img').click(function () {
        enlarge(this);
    })

    function enlarge(obj) {

        var _this = $(obj);
        imgShow("#outerdiv", "#innerdiv", "#bigimg", _this);


        function imgShow(outerdiv, innerdiv, bigimg, _this) {
            var src = _this.attr("src"); //get the src attribution
            bigImage.setAttribute("src", src);


            $("<img/>").attr("src", src).load(function () {
                var windowW = $(window).width(); //get window width
                var windowH = $(window).height();
                var realWidth = this.width; //get the real width of the image
                var realHeight = this.height;
                var imgWidth, imgHeight;
                var scale = 0.8;

                if (realHeight > windowH * scale) {
                    imgHeight = windowH * scale;
                    imgWidth = imgHeight / realHeight * realWidth;
                    if (imgWidth > windowW * scale) {
                        imgWidth = windowW * scale;
                    }
                } else if (realWidth > windowW * scale) {
                    imgWidth = windowW * scale;
                    imgHeight = imgWidth / realWidth * realHeight;
                } else {
                    imgWidth = realWidth;
                    imgHeight = realHeight;
                }
                $(bigimg).css("width", imgWidth);

                var w = (windowW - imgWidth) / 2;
                var h = (windowH - imgHeight) / 2;
                $(innerdiv).css({
                    "top": h,
                    "left": w
                });
                $(outerdiv).fadeIn("fast");
            });

            $(outerdiv).click(function () {
                $(this).fadeOut("fast");
            });
        }
    }



})

