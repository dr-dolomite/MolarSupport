"use client";

import { useRef, useState, useTransition } from "react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/button";
import { RxMagnifyingGlass } from "react-icons/rx";
import { Card } from "@/components/ui/card";
import { FaRegTrashCan } from "react-icons/fa6";

import ErrorModal from "@/components/modals/error";
import SuccessModal from "@/components/modals/success";

const M3InputCard = () => {
  const router = useRouter();
  const [dragActive, setDragActive] = useState<boolean>(false);
  const inputRef = useRef<any>(null);
  const [files, setFiles] = useState<any>([]);
  const [waitingForFile, showWaitingForFile] = useState<boolean>(true);
  const [fileUploaded, showFileUploaded] = useState<boolean>(false);
  const [showErorModal, setShowErrorModal] = useState<boolean>(false);
  const [showSuccessModal, setShowSuccessModal] = useState<boolean>(false);
  const [showLoading, setShowLoadingIcon] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string>("");
  const [isPending, startTransition] = useTransition();

  function handleChange(e: any) {
    e.preventDefault();
    console.log("File has been added");

    async function checkMcInput() {
      const formData = new FormData();
      formData.append("fileb", e.target.files[0]);
      const response = await fetch("http://127.0.0.1:8000/api/check_m3_input", {
        method: "POST",
        body: formData,
      });
      const data = await response.json();
      if (data.error) {
        setErrorMessage("The image uploaded is not a CBCT M3 slice image");
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

  // Call Start Process POST API and redirect to "/assessment" page
  async function startProcess() {
    const response = await fetch("http://127.0.0.1:8000/api/start_process", {
      method: "POST",
    });
    const data = await response.json();

    // get the "session_id" from the JSON response
    const sessionId = data.session_id;

    if (data.error) {
      setErrorMessage("Something went wrong");
      setShowErrorModal(true);
    } else {
      // Use startTransition to delay the navigation until the UI updates
      startTransition(async () => {
        // TODO: Pass the session id to ResultPage prop

        router.push(`/result/${sessionId}`);
      });
    }
  }

  function handleSubmitFile() {
    if (files.length === 0) {
      // no file has been submitted
      console.log("No file has been submitted");
      setErrorMessage(
        "No file has been submitted. Please upload the neccessary files."
      );
      setShowSuccessModal(false);
      setShowErrorModal(true);
    } else {
      // write submit logic here
      console.log("Submit button has been clicked");
      setShowLoadingIcon(true);
      startProcess();
    }
  }

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

  function showSuccessModalFunction() {
    setShowSuccessModal(true);
  }

  function closeSuccessModal() {
    setShowSuccessModal(false);
  }

  return (
    <>
      {showSuccessModal && (
        <SuccessModal onBack={closeSuccessModal} onSubmit={handleSubmitFile} showLoadingIcon={showLoading} />
      )}
      {showErorModal && (
        <ErrorModal onClose={closeErrorModal} error={errorMessage} />
      )}
      {}
      <div className="rounded-[16px] box p-8 py-12 gap-y-16 flex flex-col justify-center items-center">
        <form
          onDragEnter={handleDragEnter}
          onSubmit={(e) => e.preventDefault()}
          onDrop={handleDrop}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
        >
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
              className="flex flex-col justify-center items-center gap-y-8"
              onClick={openFileExplorer}
            >
              <img
                src="/logo/input-logo.svg"
                alt="input logo"
                className="size-24 cursor-pointer"
              />
              <h1 className="text-[#878787] px-8 text-3xl font-bold text-center cursor-pointer">
                Drag & drop to upload the CBCT M3 Axial slice image
              </h1>
              <div className="my-20">
                <p className="text-[#878787] font-medium text-xl text-center capitalize">
                  No Files Selected
                </p>
              </div>
            </div>
          )}

          {fileUploaded && (
            <>
              {files.map((file: any) => (
                <div
                  key={file.name}
                  className="flex flex-col gap-y-8 items-center justify-center px-12"
                >
                  {/* Show the uploaded image */}
                  <img
                    src={URL.createObjectURL(file)}
                    alt="uploaded file"
                    className="size-64 object-cover rounded-lg shadow-md drop-shadow-lg"
                  />

                  <Card className="flex flex-row flex-auto gap-x-12 items-center px-6 py-4">
                    <div className="flex flex-col flex-auto gap-y-1">
                      <p className="text-md font-medium truncate w-52">
                        {" "}
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
                  </Card>
                </div>
              ))}
            </>
          )}
        </form>

        <Button
          size="submitButton"
          variant="submitButton"
          className="w-full"
          onClick={showSuccessModalFunction}
        >
          <RxMagnifyingGlass className="mr-2 size-8" /> Start Assessment
        </Button>
      </div>
    </>
  );
};

export default M3InputCard;
