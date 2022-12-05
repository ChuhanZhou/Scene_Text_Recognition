
var file = document.getElementById('file');
var image = document.querySelector("#img");
var text = document.querySelector('#text')
var button = document.getElementById("btn");
var bigImage = document.querySelector('#bigimg');
var fileD;

file.onchange = function () {
        var fileData = this.files[0];//获取到一个FileList对象中的第一个文件( File 对象),是我们上传的文件
        var pettern = /^image/;

        //console.info(fileData.type)

        if (!pettern.test(fileData.type)) {
            alert("the format of picture is not correct");
            return;
        }
        var reader = new FileReader();
        reader.readAsDataURL(fileData);//异步读取文件内容，结果用data:url的字符串形式表示
        /*当读取操作成功完成时调用*/
        reader.onload = function (e) {
            //console.log(e); //查看对象
            console.log(this.result);//要的数据 这里的this指向FileReader（）对象的实例reader
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
                $.ajax({
                    url: '/upload',
                    type: 'POST',
                    data: formFile,
                    processData: false,
                    contentType: false,
                    success: function (data) {
                        console.log("post successfully");
                        text.value = data.text;
                        image.setAttribute("src", data.outPath);

                        image.style.display = "block";
                        btn.disabled = false;
                        btn.style.pointerEvents = "auto";
                        downloadBtn.disabled = false;
                        copyBtn.disabled = false;
                        btn.style.display = "block";
                        load.style.display = "none";
                        file.style.display = "block";

                        //显示文本检测结果 文件相对路径./out/r.png
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
            var src = _this.attr("src"); //获取当前点击的pimg元素中的src属性
            bigImage.setAttribute("src", src); //设置#bigimg元素的src属性

            /*获取当前点击图片的真实大小，并显示弹出层及大图*/
            $("<img/>").attr("src", src).load(function () {
                var windowW = $(window).width(); //获取当前窗口宽度
                var windowH = $(window).height(); //获取当前窗口高度
                var realWidth = this.width; //获取图片真实宽度
                var realHeight = this.height; //获取图片真实高度
                var imgWidth, imgHeight;
                var scale = 0.8; //缩放尺寸，当图片真实宽度和高度大于窗口宽度和高度时进行缩放

                if (realHeight > windowH * scale) { //判断图片高度
                    imgHeight = windowH * scale; //如大于窗口高度，图片高度进行缩放
                    imgWidth = imgHeight / realHeight * realWidth; //等比例缩放宽度
                    if (imgWidth > windowW * scale) { //如宽度扔大于窗口宽度
                        imgWidth = windowW * scale; //再对宽度进行缩放
                    }
                } else if (realWidth > windowW * scale) { //如图片高度合适，判断图片宽度
                    imgWidth = windowW * scale; //如大于窗口宽度，图片宽度进行缩放
                    imgHeight = imgWidth / realWidth * realHeight; //等比例缩放高度
                } else { //如果图片真实高度和宽度都符合要求，高宽不变
                    imgWidth = realWidth;
                    imgHeight = realHeight;
                }
                $(bigimg).css("width", imgWidth); //以最终的宽度对图片缩放

                var w = (windowW - imgWidth) / 2; //计算图片与窗口左边距
                var h = (windowH - imgHeight) / 2; //计算图片与窗口上边距
                $(innerdiv).css({
                    "top": h,
                    "left": w
                }); //设置#innerdiv的top和left属性
                $(outerdiv).fadeIn("fast"); //淡入显示#outerdiv及.pimg
            });

            $(outerdiv).click(function () { //再次点击淡出消失弹出层
                $(this).fadeOut("fast");
            });
        }
    }



})

