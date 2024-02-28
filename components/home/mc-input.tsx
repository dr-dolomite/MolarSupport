"use client";

import { Card } from "@/components/ui/card";
import { FaArrowUpFromBracket } from "react-icons/fa6";
import { useState, useRef } from "react";
import { FaRegTrashCan } from "react-icons/fa6";
import ErrorModal from "@/components/modals/error";

const McInputCard = () => {
  const [dragActive, setDragActive] = useState<boolean>(false);
  const inputRef = useRef<any>(null);
  const [files, setFiles] = useState<any>([]);
  const [waitingForFile, showWaitingForFile] = useState<boolean>(true);
  const [fileUploaded, showFileUploaded] = useState<boolean>(false);
  const [showErorModal, setShowErrorModal] = useState<boolean>(false);

  function handleChange(e: any) {
    e.preventDefault();
    console.log("File has been added");

    /*
     * Function to check if the uploaded file is a valid CBCT MC slice
     * @returns either success or error message
     */

    async function checkMcInput() {
      const formData = new FormData();
      formData.append("fileb", e.target.files[0]);
      const response = await fetch("http://127.0.0.1:8000/api/check_mc_input", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        setShowErrorModal(true);
      }
    }

    // check if the file is a valid CBCT MC slice
    checkMcInput();
    showFileUploaded(true);
    showWaitingForFile(false);

    if (e.target.files && e.target.files[0]) {
      console.log(e.target.files);
      for (let i = 0; i < e.target.files["length"]; i++) {
        setFiles((prevState: any) => [...prevState, e.target.files[i]]);
      }
    }
  }

  // function handleSubmitFile(e: any) {
  //   if (files.length === 0) {
  //     // no file has been submitted
  //   } else {
  //     // write submit logic here
  //     //   showWaitingForFile(true);
  //     //   showFileUploaded(false);
  //   }
  // }

  function handleDrop(e: any) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const droppedFiles = Array.from(e.dataTransfer.files);
      setFiles((prevState: any) => [...prevState, ...droppedFiles]);
      showFileUploaded(true);
      showWaitingForFile(false);
    }
  }

  function handleDragLeave(e: any) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }

  function handleDragOver(e: any) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }

  function handleDragEnter(e: any) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }

  function removeFile(fileName: any) {
    const filteredFiles = files.filter((file: any) => file.name !== fileName);
    setFiles(filteredFiles);
    showWaitingForFile(true);
    showFileUploaded(false);
  }

  function openFileExplorer() {
    inputRef.current.value = "";
    inputRef.current.click();
  }

  // Function to calculate file size in MB
  function formatBytes(bytes: number, decimals: number = 2) {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const dm = decimals < 0 ? 0 : decimals;
    const sizes = ["Bytes", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + " " + sizes[i];
  }

  // Add a function to handle the error modal
  // When Try Again button is clicked, the modal should close and the image will be removed
  function closeErrorModal() {
    setShowErrorModal(false);
    setFiles([]);
    showWaitingForFile(true);
    showFileUploaded(false);
  }

  return (
    <>
      {showErorModal && (
        <ErrorModal
          onClose={closeErrorModal}
          error="Image upload was not 
CBCT MC slice Image"
        />
      )}

      <form
        onDragEnter={handleDragEnter}
        onSubmit={(e) => e.preventDefault()}
        onDrop={handleDrop}
        onDragLeave={handleDragLeave}
        onDragOver={handleDragOver}
      >
        <Card className="bg-[#D7CEFF] hover:bg-[#D7CEFF]/80 border-2 border-[#6D58C6] py-6 px-8 cursor-pointer">
          {/* this input element allows us to select files for upload. We make it hidden so we can activate it when the user clicks select files */}
          <input
            placeholder="fileInput"
            className="hidden"
            ref={inputRef}
            type="file"
            multiple={false}
            onChange={handleChange}
            accept="image/*"
          />
          {waitingForFile && (
            <div
              className="flex flex-row justify-center items-center gap-x-4 cursor-pointer text-[#6D58C6]"
              onClick={openFileExplorer}
            >
              <FaArrowUpFromBracket className="text-3xl" />
              <p className="text-2xl font-medium">
                Upload CBCT Mandibular Canal
              </p>
            </div>
          )}

          {fileUploaded && (
            <>
              {files.map((file: any) => (
                <Card
                  key={file.name}
                  className="flex flex-row gap-x-4 items-center px-6 py-4"
                >
                  {/* Show the uploaded image */}
                  <img
                    src={URL.createObjectURL(file)}
                    alt="uploaded file"
                    className="size-24 object-cover rounded-lg shadow-md drop-shadow-lg"
                  />

                  <div className="flex flex-row flex-auto gap-x-4 items-center px-6 py-4">
                    <div className="flex flex-col flex-auto gap-y-1">
                      <p className="text-md font-medium truncate w-52">
                        {file.name}
                      </p>
                      <p className="text-[#929292] text-sm font-medium leading-tight">
                        {formatBytes(file.size)}
                      </p>
                    </div>

                    <FaRegTrashCan
                      className="text-2xl text-red-400 hover:text-red-400/80 cursor-pointer"
                      onClick={() => removeFile(file.name)}
                    />
                  </div>
                </Card>
              ))}
            </>
          )}
        </Card>
      </form>
    </>
  );
};

export default McInputCard;
