import { MdClose } from "react-icons/md";
import { TiWarningOutline } from "react-icons/ti";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
  
  interface DeleteModalProps {
    onClose: () => void;
    deleteDatabase: () => void;
    disableButton: boolean;
  }
  
  export const DeleteModal = ({
    onClose,
    deleteDatabase,
    disableButton,
  }: DeleteModalProps) => {
    return (
      <div className="fixed top-0 left-0 right-0 bottom-0 backdrop-blur-sm backdrop-brightness-[30%] z-50 flex items-center justify-center">
        <Card>
          <Button variant="ghost" size="ghost" className="mt-8 ml-4 mb-0">
            <MdClose className="text-4xl" onClick={onClose} />
          </Button>
  
          <div className="flex flex-col p-8 justify-center items-center">
            <TiWarningOutline className="text-[#FF4D4F] size-60" />
            <h1 className="font-semibold text-[#667085] text-[1.5rem] mt-12">
              Do you want to delete all of the stored sessions?
            </h1>
  
            <Button
              variant="destructiveButton"
              size="purpleButton"
              className="mt-8 w-full"
              onClick={deleteDatabase}
              disabled={disableButton}
            >
              Delete
            </Button>
          </div>
        </Card>
      </div>
    );
  };
  