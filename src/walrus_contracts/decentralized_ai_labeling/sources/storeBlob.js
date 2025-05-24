function storeBlob() {
  const inputFile = document.getElementById("file-input").files[0];
  const numEpochs = document.getElementById("epochs-input").value;
  const basePublisherUrl = document.getElementById("publisher-url-input").value;

  // 這裡填入你的 Sui 地址
  let sendTo = document.getElementById("send-to-input").value.trim();
  let sendToParam = sendTo ? `&send_object_to=${sendTo}`: "";
  return fetch(`${basePublisherUrl}/v1/blobs?epochs=${numEpochs}${sendToParam}`, {
    method: "PUT",
    body: inputFile,
  }).then((response) => {
    if (response.status === 200) {
      return response.json().then((info) => {
        console.log(info);
        return { info: info, media_type: inputFile.type };
      });
    } else {
      throw new Error("Something went wrong when storing the blob!");
    }
  })
}