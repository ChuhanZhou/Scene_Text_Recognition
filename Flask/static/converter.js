
var file = document.getElementById('file');
var image = document.querySelector("img");
var text = document.querySelector('#text')

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
            image.setAttribute("src", this.result)
            image.style.display = "block"
            fileD = this.result
            console.log("fileD is " + fileD)
        }


    }

$(document).ready(function() {

    $('#btn').click(function () {
        console.log('click button >>>>')
            if (fileD!=="" && fileD!=null)
            {
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
                        text.value = data;
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

})

