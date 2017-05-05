#ifndef TH_GENERIC_FILE
#define TH_GENERIC_FILE "master_worker/master/generic/THDTensorCopy.cpp"
#else

// TODO implement
void THDTensor_(copy)(THDTensor *tensor, THDTensor *src) {
  throw std::runtime_error("copy not implemented yet");
}

void THDTensor_(copyFromMaster)(THDTensor* to, THDTensorDescriptor* from) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCopyFromMaster, to),
    THDState::s_current_worker
  );

  thd::dataChannel->send(*from, THDState::s_current_worker);
}

void THDTensor_(copyFromWorker)(THDTensorDescriptor* to, THDTensor* from) {
  masterCommandChannel->sendMessage(
    packMessage(Functions::tensorCopyFromWorker, from),
    THDState::s_current_worker
  );

  thd::dataChannel->receive(*to, THDState::s_current_worker);
}

#endif
